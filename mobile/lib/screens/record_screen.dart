import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:audioplayers/audioplayers.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import '../main.dart' show AppColors, appCardShadow;
import '../services/api_service.dart';
import '../services/audio_recorder_service.dart';
import '../services/recordings_store.dart';
import '../widgets/enhance_speech_card.dart';
import 'recordings_screen.dart';

enum _State { idle, recording, processing, ready }

// ─────────────────────────────────────────────────────────────────────────────
//  TOP-LEVEL SCREEN
// ─────────────────────────────────────────────────────────────────────────────
class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});
  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  bool _isRecordTab = true;

  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.of(context).padding.top;
    return Scaffold(
      backgroundColor: AppColors.background,
      body: Column(crossAxisAlignment: CrossAxisAlignment.center, children: [
        SizedBox(height: pad + 16),
        Center(child: _Toggle(
          isLeft: _isRecordTab,
          onToggle: (v) => setState(() => _isRecordTab = v))),
        const SizedBox(height: 24),
        Expanded(child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 200),
          child: _isRecordTab
              ? const _RecordTab(key: ValueKey('rec'))
              : const _UploadTab(key: ValueKey('up')))),
      ]),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  RECORD TAB
// ─────────────────────────────────────────────────────────────────────────────
class _RecordTab extends StatefulWidget {
  const _RecordTab({super.key});
  @override
  State<_RecordTab> createState() => _RecordTabState();
}

class _RecordTabState extends State<_RecordTab> {
  final _recorder = AudioRecorderService();
  final _player   = AudioPlayer();

  _State   _state = _State.idle;
  String?  _enhancedPath;
  String?  _rawPath;
  String   _msg = '';
  Duration _elapsed = Duration.zero;

  bool   _noiseReductionEnabled = true;
  double _noiseStrength = 1.0;
  bool   _reprocessing = false;

  Timer? _clock;
  StreamSubscription<Amplitude>? _ampSub;

  static const int _kBars = 60;
  final List<double> _bars = List.filled(_kBars, 0.0);

  // Minimum file size — lower on Windows because WAV headers alone are ~44 bytes
  // and short recordings may be small but still valid
  static int get _minBytes => Platform.isWindows ? 512 : 4096;

  @override
  void dispose() {
    _clock?.cancel();
    _ampSub?.cancel();
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  // ── Start ─────────────────────────────────────────────────────────────────
  Future<void> _start() async {
    final hasPerm = await _recorder.requestPermission();
    if (!hasPerm) {
      setState(() => _msg =
        'Microphone permission denied.\n'
        'Windows: Settings → Privacy → Microphone → allow this app.');
      return;
    }

    final dir = await getTemporaryDirectory();
    try {
      _rawPath = await _recorder.start(dir.path);
      debugPrint('[Record] writing to $_rawPath');
    } catch (e) {
      debugPrint('[Record] start failed: $e');
      setState(() => _msg = '$e');
      return;
    }

    setState(() {
      _state = _State.recording;
      _msg = '';
      _enhancedPath = null;
      _elapsed = Duration.zero;
      _noiseReductionEnabled = true;
      _noiseStrength = 1.0;
      for (int i = 0; i < _kBars; i++) _bars[i] = 0;
    });
    _startClock();
    _startWave();
  }

  // ── Stop + send to server ─────────────────────────────────────────────────
  Future<void> _stop() async {
    _clock?.cancel();
    await _ampSub?.cancel();
    _ampSub = null;

    final path = await _recorder.stop();
    debugPrint('[Record] stopped, path=$path');

    if (path == null) {
      setState(() { _state = _State.idle; _msg = 'Recording path was null.'; });
      return;
    }

    final fileExists = await File(path).exists();
    final fileSize   = fileExists ? await File(path).length() : 0;
    debugPrint('[Record] file exists=$fileExists  size=$fileSize B  min=$_minBytes B');

    if (!fileExists || fileSize < _minBytes) {
      setState(() {
        _state = _State.idle;
        _msg = 'Recording too small ($fileSize B). '
               'Check Windows Settings → System → Sound → Input '
               'and make sure your mic is the default input device.';
      });
      return;
    }

    setState(() { _state = _State.processing; _msg = 'Enhancing on server…'; });

    // Explicit, not relying on any persisted default — always start a fresh
    // recording at full noise-reduction strength.
    final bytes = await ApiService.filterFile(File(path), noiseStrength: _noiseStrength);
    if (bytes == null || bytes.isEmpty) {
      setState(() {
        _state = _State.idle;
        _msg = 'Server returned no audio. Check Settings → Server URL.';
      });
      return;
    }

    // Save to permanent storage
    final docs = await getApplicationDocumentsDirectory();
    final out  = '${docs.path}/enhanced_${DateTime.now().millisecondsSinceEpoch}.wav';
    await File(out).writeAsBytes(bytes, flush: true);
    debugPrint('[Record] enhanced WAV: $out (${bytes.length} B)');

    // Persist metadata
    final rec = Recording(
      id:           DateTime.now().millisecondsSinceEpoch.toString(),
      name:         'New Recording',
      enhancedPath: out,
      originalPath: _rawPath,
      createdAt:    DateTime.now(),
      durationSec:  _elapsed.inSeconds.toDouble(),
    );
    await RecordingsStore.add(rec);

    setState(() {
      _state        = _State.ready;
      _enhancedPath = out;
      _msg          = 'Done! Tap Play to listen.';
    });
  }

  // ── Re-process the same raw recording with new Enhance Speech settings ─────
  Future<void> _reprocess() async {
    if (_rawPath == null) return;
    setState(() => _reprocessing = true);

    final effective = _noiseReductionEnabled ? _noiseStrength : 0.0;
    final bytes = await ApiService.filterFile(File(_rawPath!), noiseStrength: effective);
    if (!mounted) return;

    if (bytes == null || bytes.isEmpty) {
      setState(() {
        _reprocessing = false;
        _msg = 'Re-processing failed: ${ApiService.lastError ?? 'unknown error'}';
      });
      return;
    }

    final docs = await getApplicationDocumentsDirectory();
    final out  = '${docs.path}/enhanced_${DateTime.now().millisecondsSinceEpoch}.wav';
    await File(out).writeAsBytes(bytes, flush: true);
    debugPrint('[Record] re-enhanced WAV: $out (${bytes.length} B)');

    if (!mounted) return;
    setState(() {
      _reprocessing = false;
      _enhancedPath = out;
    });
  }

  // ── Playback ──────────────────────────────────────────────────────────────
  Future<void> _play(String path) async {
    if (!await File(path).exists()) {
      setState(() => _msg = 'File missing: $path');
      return;
    }
    await _player.stop();
    await _player.play(DeviceFileSource(path));
  }

  // ── Clock ─────────────────────────────────────────────────────────────────
  void _startClock() {
    _clock = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted && _state == _State.recording) {
        setState(() => _elapsed += const Duration(seconds: 1));
      }
    });
  }

  // ── Waveform ──────────────────────────────────────────────────────────────
  Future<void> _startWave() async {
    final stream = await _recorder.amplitudeStream();
    _ampSub?.cancel();
    _ampSub = stream.listen((amp) {
      if (!mounted) return;
      final n = ((amp.current + 50.0) / 50.0).clamp(0.0, 1.0).toDouble();
      setState(() {
        for (int i = 0; i < _kBars - 1; i++) _bars[i] = _bars[i + 1];
        _bars[_kBars - 1] = n;
      });
    },
    onError: (e) => debugPrint('[Record] amp error: $e'),
    cancelOnError: false);
  }

  String get _timer {
    final h = _elapsed.inHours.toString().padLeft(2, '0');
    final m = _elapsed.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = _elapsed.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$h:$m:$s';
  }

  // ── UI ────────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 110),
      child: Column(children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
          decoration: BoxDecoration(color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: appCardShadow()),
          child: Column(children: [
            _Pill(state: _state),
            const SizedBox(height: 20),
            SizedBox(height: 64, child: _Wave(bars: _bars)),
            const SizedBox(height: 16),
            Text(_timer, style: const TextStyle(
                fontSize: 28, fontWeight: FontWeight.w700, letterSpacing: 2)),
          ])),

        const SizedBox(height: 20),
        if (_msg.isNotEmpty)
          Padding(padding: const EdgeInsets.only(bottom: 16),
            child: Text(_msg, textAlign: TextAlign.center,
                style: TextStyle(
                    color: _msg.startsWith('Done') ? AppColors.green
                         : AppColors.textSecondary,
                    fontSize: 13.5))),

        if (_state == _State.processing)
          const LinearProgressIndicator(minHeight: 4,
              backgroundColor: Color(0xFFFEE2E2), color: AppColors.red),

        const SizedBox(height: 16),
        _primaryButton(),

        if (_state == _State.ready && _enhancedPath != null) ...[
          const SizedBox(height: 16),
          EnhanceSpeechCard(
            enabled: _noiseReductionEnabled,
            strength: _noiseStrength,
            busy: _reprocessing,
            onEnabledChanged: (v) {
              setState(() => _noiseReductionEnabled = v);
              _reprocess();
            },
            onStrengthChanged: (v) => setState(() => _noiseStrength = v),
            onStrengthChangeEnd: (v) {
              setState(() => _noiseStrength = v);
              _reprocess();
            },
          ),
          const SizedBox(height: 12),
          _Button(label: 'Play Enhanced',
              icon: Icons.play_arrow_rounded,
              color: AppColors.green,
              onTap: () => _play(_enhancedPath!)),
          const SizedBox(height: 10),
          _Button(
              label: 'View All Recordings',
              icon:  Icons.list_alt_rounded,
              color: AppColors.red,
              outline: true,
              onTap: () => Navigator.push(context,
                  MaterialPageRoute(builder: (_) => const RecordingsScreen()))),
          const SizedBox(height: 10),
          if (_rawPath != null)
            _Button(label: 'Play Original (Compare)',
                icon: Icons.compare_arrows_rounded,
                color: AppColors.red,
                outline: true,
                onTap: () => _play(_rawPath!)),
          const SizedBox(height: 10),
          _Button(label: 'Record Again',
              icon: Icons.mic_rounded,
              color: AppColors.red,
              outline: true,
              onTap: _start),
        ],
      ]),
    );
  }

  Widget _primaryButton() {
    switch (_state) {
      case _State.idle:
      case _State.ready:
        return _Button(label: 'Start Recording',
            icon: Icons.mic_rounded, color: AppColors.red, onTap: _start);
      case _State.recording:
        return _Button(label: 'Stop',
            icon: Icons.stop_rounded, color: AppColors.red, onTap: _stop);
      case _State.processing:
        return const SizedBox(height: 56);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  UPLOAD TAB
// ─────────────────────────────────────────────────────────────────────────────
class _UploadTab extends StatefulWidget {
  const _UploadTab({super.key});
  @override
  State<_UploadTab> createState() => _UploadTabState();
}

class _UploadTabState extends State<_UploadTab> {
  final _player = AudioPlayer();
  PlatformFile? _file;
  bool _busy = false;
  String _msg = '';
  String? _outPath;

  bool   _noiseReductionEnabled = true;
  double _noiseStrength = 1.0;
  bool   _reprocessing = false;

  Future<void> _pick() async {
    final r = await FilePicker.platform
        .pickFiles(type: FileType.audio, allowMultiple: false);
    if (r != null && mounted) {
      setState(() {
        _file = r.files.single; _outPath = null; _msg = '';
        _noiseReductionEnabled = true; _noiseStrength = 1.0;
      });
    }
  }

  Future<void> _filter() async {
    if (_file?.path == null) return;
    setState(() { _busy = true; _msg = 'Sending to server…'; });
    final bytes = await ApiService.filterFile(File(_file!.path!), noiseStrength: _noiseStrength);
    if (bytes == null || bytes.isEmpty) {
      setState(() { _busy = false;
          _msg = 'Server returned nothing. Check Settings → Server URL.'; });
      return;
    }
    final dir = await getTemporaryDirectory();
    final out = '${dir.path}/upload_${DateTime.now().millisecondsSinceEpoch}.wav';
    await File(out).writeAsBytes(bytes, flush: true);
    setState(() { _busy = false; _outPath = out; _msg = 'Done! Tap Play.'; });
  }

  // ── Re-process the same picked file with new Enhance Speech settings ───────
  Future<void> _reprocess() async {
    if (_file?.path == null) return;
    setState(() => _reprocessing = true);

    final effective = _noiseReductionEnabled ? _noiseStrength : 0.0;
    final bytes = await ApiService.filterFile(File(_file!.path!), noiseStrength: effective);
    if (!mounted) return;

    if (bytes == null || bytes.isEmpty) {
      setState(() {
        _reprocessing = false;
        _msg = 'Re-processing failed: ${ApiService.lastError ?? 'unknown error'}';
      });
      return;
    }

    final dir = await getTemporaryDirectory();
    final out = '${dir.path}/upload_${DateTime.now().millisecondsSinceEpoch}.wav';
    await File(out).writeAsBytes(bytes, flush: true);

    if (!mounted) return;
    setState(() { _reprocessing = false; _outPath = out; });
  }

  Future<void> _play() async {
    if (_outPath == null) return;
    await _player.stop();
    await _player.play(DeviceFileSource(_outPath!));
  }

  @override
  void dispose() { _player.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) => SingleChildScrollView(
    padding: const EdgeInsets.fromLTRB(20, 0, 20, 110),
    child: Column(children: [
      GestureDetector(
        onTap: _busy ? null : _pick,
        child: Container(width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 34),
          decoration: BoxDecoration(color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: appCardShadow()),
          child: Column(children: [
            Container(width: 64, height: 64,
              decoration: BoxDecoration(color: AppColors.redLight,
                  borderRadius: BorderRadius.circular(16)),
              child: const Icon(Icons.graphic_eq_rounded,
                  color: AppColors.red, size: 32)),
            const SizedBox(height: 20),
            if (_file != null)
              Padding(padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Text(_file!.name, textAlign: TextAlign.center,
                    maxLines: 2, overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontSize: 13.5,
                        fontWeight: FontWeight.w500))),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
              decoration: BoxDecoration(color: AppColors.red,
                  borderRadius: BorderRadius.circular(30)),
              child: Text(_file == null ? 'Pick Audio File' : 'Change File',
                  style: const TextStyle(color: Colors.white,
                      fontWeight: FontWeight.w600, fontSize: 15))),
          ]))),
      if (_busy) ...[
        const SizedBox(height: 12),
        const LinearProgressIndicator(minHeight: 4,
            backgroundColor: Color(0xFFFEE2E2), color: AppColors.red),
      ],
      if (_msg.isNotEmpty) ...[
        const SizedBox(height: 12),
        Text(_msg, textAlign: TextAlign.center,
            style: TextStyle(
                color: _msg.startsWith('Done') ? AppColors.green
                     : AppColors.textSecondary, fontSize: 13.5)),
      ],
      if (_file != null && !_busy && _outPath == null) ...[
        const SizedBox(height: 20),
        _Button(label: 'Filter Audio', icon: Icons.auto_fix_high,
            color: AppColors.red, onTap: _filter),
      ],
      if (_outPath != null) ...[
        const SizedBox(height: 20),
        EnhanceSpeechCard(
          enabled: _noiseReductionEnabled,
          strength: _noiseStrength,
          busy: _reprocessing,
          onEnabledChanged: (v) {
            setState(() => _noiseReductionEnabled = v);
            _reprocess();
          },
          onStrengthChanged: (v) => setState(() => _noiseStrength = v),
          onStrengthChangeEnd: (v) {
            setState(() => _noiseStrength = v);
            _reprocess();
          },
        ),
        const SizedBox(height: 12),
        _Button(label: 'Play Filtered Audio', icon: Icons.play_arrow_rounded,
            color: AppColors.green, onTap: _play),
        const SizedBox(height: 10),
        _Button(label: 'Pick Another File', icon: Icons.refresh_rounded,
            color: AppColors.red, outline: true, onTap: _pick),
      ],
    ]),
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Shared widgets
// ─────────────────────────────────────────────────────────────────────────────
class _Pill extends StatelessWidget {
  final _State state;
  const _Pill({required this.state});
  @override
  Widget build(BuildContext context) {
    final (label, active) = switch (state) {
      _State.idle       => ('Ready',       false),
      _State.recording  => ('Recording',   true ),
      _State.processing => ('Processing…', true ),
      _State.ready      => ('Done',        false),
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
      decoration: BoxDecoration(color: AppColors.redLight,
          borderRadius: BorderRadius.circular(20)),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        AnimatedContainer(duration: const Duration(milliseconds: 250),
            width: 8, height: 8,
            decoration: BoxDecoration(
                color: active ? AppColors.red : const Color(0xFFD1D5DB),
                shape: BoxShape.circle)),
        const SizedBox(width: 8),
        Text(label, style: const TextStyle(
            color: AppColors.red, fontWeight: FontWeight.w500, fontSize: 13)),
      ]),
    );
  }
}

class _Wave extends StatelessWidget {
  final List<double> bars;
  const _Wave({required this.bars});
  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (_, c) {
      final maxH = c.maxHeight;
      return Row(mainAxisAlignment: MainAxisAlignment.center,
        children: List.generate(bars.length, (i) {
          final h = (4 + math.pow(bars[i], 0.5) * (maxH - 4)).toDouble();
          return Padding(padding: const EdgeInsets.symmetric(horizontal: 1),
            child: AnimatedContainer(duration: const Duration(milliseconds: 50),
                width: 3, height: h,
                decoration: BoxDecoration(color: AppColors.red,
                    borderRadius: BorderRadius.circular(2))));
        }));
    });
  }
}

class _Button extends StatelessWidget {
  final String label;
  final IconData? icon;
  final Color color;
  final bool outline;
  final VoidCallback onTap;
  const _Button({required this.label, this.icon, required this.color,
      required this.onTap, this.outline = false});

  @override
  Widget build(BuildContext context) {
    final child = Row(mainAxisAlignment: MainAxisAlignment.center, children: [
      if (icon != null) ...[Icon(icon, size: 20), const SizedBox(width: 8)],
      Text(label, style: const TextStyle(
          fontSize: 15.5, fontWeight: FontWeight.w600)),
    ]);
    return SizedBox(width: double.infinity, height: 56,
      child: outline
        ? OutlinedButton(onPressed: onTap,
            style: OutlinedButton.styleFrom(
              foregroundColor: color,
              side: BorderSide(color: color, width: 1.5),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30))),
            child: child)
        : ElevatedButton(onPressed: onTap,
            style: ElevatedButton.styleFrom(
              backgroundColor: color, foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30))),
            child: child));
  }
}

class _Toggle extends StatelessWidget {
  final bool isLeft;
  final ValueChanged<bool> onToggle;
  const _Toggle({required this.isLeft, required this.onToggle});

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(5),
    decoration: BoxDecoration(color: Colors.white,
        borderRadius: BorderRadius.circular(30),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.07),
            blurRadius: 12, offset: const Offset(0, 2))]),
    child: Row(mainAxisSize: MainAxisSize.min, children: [
      _pill('Record', isLeft,  () => onToggle(true)),
      _pill('Upload', !isLeft, () => onToggle(false)),
    ]));

  Widget _pill(String s, bool active, VoidCallback onTap) =>
    GestureDetector(onTap: onTap,
      child: AnimatedContainer(duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 11),
        decoration: BoxDecoration(
            color: active ? AppColors.red : Colors.transparent,
            borderRadius: BorderRadius.circular(25)),
        child: Text(s, style: TextStyle(
            color: active ? Colors.white : AppColors.textHint,
            fontWeight: FontWeight.w600, fontSize: 14.5))));
}