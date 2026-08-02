import 'dart:async';
import 'dart:io';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';

import '../main.dart' show AppColors, appCardShadow;
import '../services/recordings_store.dart';

class RecordingsScreen extends StatefulWidget {
  const RecordingsScreen({super.key});
  @override
  State<RecordingsScreen> createState() => _RecordingsScreenState();
}

class _RecordingsScreenState extends State<RecordingsScreen> {
  List<Recording> _list = [];
  String?         _activeId;
  bool            _selectMode = false;
  final Set<String> _selected = {};

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final l = await RecordingsStore.loadAll();
    if (mounted) setState(() => _list = l);
  }

  Future<void> _delete(String id) async {
    final r = _list.firstWhere((x) => x.id == id);
    try { await File(r.enhancedPath).delete(); } catch (_) {}
    if (r.originalPath != null) {
      try { await File(r.originalPath!).delete(); } catch (_) {}
    }
    await RecordingsStore.remove(id);
    _activeId = null;
    _load();
  }

  Future<void> _rename(Recording r) async {
    final ctl = TextEditingController(text: r.name);
    final result = await showDialog<String>(context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('Rename', style: TextStyle(fontWeight: FontWeight.w700)),
        content: TextField(controller: ctl, autofocus: true,
            decoration: const InputDecoration(hintText: 'Recording name')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel')),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.red, foregroundColor: Colors.white),
            onPressed: () => Navigator.pop(ctx, ctl.text.trim()),
            child: const Text('Save')),
        ]));
    if (result != null && result.isNotEmpty) {
      await RecordingsStore.rename(r.id, result);
      _load();
    }
  }

  Future<void> _bulkDelete() async {
    for (final id in List<String>.from(_selected)) {
      await _delete(id);
    }
    setState(() {
      _selected.clear();
      _selectMode = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.of(context).padding.top;
    return Scaffold(
      backgroundColor: AppColors.background,
      body: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        // ── Top bar ────────────────────────────────────────────────────────
        SizedBox(height: pad + 12),
        Padding(padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(children: [
            _CircleIconButton(icon: Icons.arrow_back_ios_new_rounded,
              onTap: () => Navigator.pop(context)),
            const Spacer(),
            if (_selectMode && _selected.isNotEmpty)
              _CircleIconButton(icon: Icons.delete_rounded,
                color: AppColors.red, onTap: _bulkDelete),
            const SizedBox(width: 8),
            _PillButton(
              label: _selectMode ? 'Done' : 'Select',
              onTap: () => setState(() {
                _selectMode = !_selectMode;
                _selected.clear();
              })),
          ])),

        // ── Title ──────────────────────────────────────────────────────────
        Padding(padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
          child: const Text('All Recordings',
            style: TextStyle(fontSize: 30, fontWeight: FontWeight.w800,
                color: AppColors.textPrimary, letterSpacing: -0.5))),
        Padding(padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Container(height: 1, color: AppColors.border)),

        // ── List ───────────────────────────────────────────────────────────
        Expanded(child: _list.isEmpty
          ? const _EmptyState()
          : ListView.separated(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
              itemCount: _list.length,
              separatorBuilder: (_, __) => const SizedBox(height: 8),
              itemBuilder: (_, i) {
                final r = _list[i];
                return _RecordingCard(
                  rec: r,
                  isActive: _activeId == r.id,
                  selectMode: _selectMode,
                  selected: _selected.contains(r.id),
                  onTap: () {
                    if (_selectMode) {
                      setState(() {
                        if (_selected.contains(r.id)) {
                          _selected.remove(r.id);
                        } else {
                          _selected.add(r.id);
                        }
                      });
                    } else {
                      setState(() =>
                        _activeId = _activeId == r.id ? null : r.id);
                    }
                  },
                  onDelete: () => _delete(r.id),
                  onRename: () => _rename(r),
                );
              })),
      ]),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Recording card with embedded player
// ─────────────────────────────────────────────────────────────────────────────
class _RecordingCard extends StatefulWidget {
  final Recording rec;
  final bool isActive;
  final bool selectMode;
  final bool selected;
  final VoidCallback onTap;
  final VoidCallback onDelete;
  final VoidCallback onRename;

  const _RecordingCard({
    required this.rec,
    required this.isActive,
    required this.selectMode,
    required this.selected,
    required this.onTap,
    required this.onDelete,
    required this.onRename,
  });

  @override
  State<_RecordingCard> createState() => _RecordingCardState();
}

class _RecordingCardState extends State<_RecordingCard> {
  final _player = AudioPlayer();
  Duration _pos      = Duration.zero;
  Duration _dur      = Duration.zero;
  bool     _playing  = false;
  StreamSubscription<Duration>? _posSub;
  StreamSubscription<Duration>? _durSub;
  StreamSubscription<void>?     _completeSub;

  @override
  void initState() {
    super.initState();
    _posSub = _player.onPositionChanged.listen((d) {
      if (mounted) setState(() => _pos = d);
    });
    _durSub = _player.onDurationChanged.listen((d) {
      if (mounted) setState(() => _dur = d);
    });
    _completeSub = _player.onPlayerComplete.listen((_) {
      if (mounted) setState(() {
        _playing = false;
        _pos = Duration.zero;
      });
    });
    _dur = Duration(milliseconds: (widget.rec.durationSec * 1000).round());
  }

  @override
  void dispose() {
    _posSub?.cancel(); _durSub?.cancel(); _completeSub?.cancel();
    _player.dispose();
    super.dispose();
  }

  Future<void> _toggle() async {
    if (_playing) {
      await _player.pause();
      setState(() => _playing = false);
    } else {
      await _player.play(DeviceFileSource(widget.rec.enhancedPath));
      setState(() => _playing = true);
    }
  }

  Future<void> _seek(Duration delta) async {
    final target = (_pos + delta).inMilliseconds.clamp(0, _dur.inMilliseconds);
    await _player.seek(Duration(milliseconds: target));
  }

  String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(1, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  String _stamp() {
    final now  = DateTime.now();
    final r    = widget.rec.createdAt;
    final today = DateTime(now.year, now.month, now.day);
    final that  = DateTime(r.year, r.month, r.day);
    final hh    = r.hour.toString().padLeft(2, '0');
    final mm    = r.minute.toString().padLeft(2, '0');
    if (that == today) return '$hh:$mm';
    final daysDiff = today.difference(that).inDays;
    if (daysDiff == 1) return 'Yesterday';
    if (daysDiff < 7)  {
      const wd = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
      return wd[r.weekday - 1];
    }
    return '${r.day}/${r.month}/${r.year}';
  }

  @override
  Widget build(BuildContext context) {
    final progress = _dur.inMilliseconds > 0
      ? _pos.inMilliseconds / _dur.inMilliseconds : 0.0;

    return GestureDetector(
      onTap: widget.onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        decoration: BoxDecoration(color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: appCardShadow(),
            border: widget.selected
              ? Border.all(color: AppColors.red, width: 2)
              : null),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            // ── Header row ──────────────────────────────────────────────
            Row(children: [
              if (widget.selectMode) ...[
                Icon(widget.selected
                    ? Icons.check_circle_rounded
                    : Icons.radio_button_unchecked_rounded,
                  color: widget.selected ? AppColors.red : AppColors.textHint),
                const SizedBox(width: 12),
              ],
              Expanded(child: Column(
                crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(widget.rec.name, maxLines: 1, overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textPrimary)),
                const SizedBox(height: 2),
                Text(_stamp(),
                  style: const TextStyle(fontSize: 13,
                      color: AppColors.textHint)),
              ])),
              if (!widget.selectMode)
                IconButton(onPressed: widget.onRename,
                  icon: const Icon(Icons.more_horiz_rounded,
                    color: AppColors.red, size: 22)),
            ]),

            // ── Progress + controls (only when expanded) ────────────────
            if (widget.isActive && !widget.selectMode) ...[
              const SizedBox(height: 12),
              ClipRRect(borderRadius: BorderRadius.circular(2),
                child: LinearProgressIndicator(
                  value:           progress.clamp(0.0, 1.0),
                  backgroundColor: AppColors.iconBg,
                  color:           AppColors.red,
                  minHeight:       4)),
              const SizedBox(height: 6),
              Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                Text(_fmt(_pos),
                  style: const TextStyle(fontSize: 12,
                    color: AppColors.textHint)),
                Text('-${_fmt(_dur - _pos)}',
                  style: const TextStyle(fontSize: 12,
                    color: AppColors.textHint)),
              ]),
              const SizedBox(height: 14),
              Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
                Icon(Icons.equalizer_rounded,
                  color: _playing ? AppColors.red : AppColors.textHint, size: 22),
                _IconBtn(icon: Icons.replay_5_rounded,
                  onTap: () => _seek(const Duration(seconds: -5))),
                _PlayBtn(playing: _playing, onTap: _toggle),
                _IconBtn(icon: Icons.forward_5_rounded,
                  onTap: () => _seek(const Duration(seconds: 5))),
                _IconBtn(icon: Icons.delete_outline_rounded,
                  onTap: widget.onDelete),
              ]),
            ],
          ])),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helper widgets
// ─────────────────────────────────────────────────────────────────────────────
class _PlayBtn extends StatelessWidget {
  final bool playing;
  final VoidCallback onTap;
  const _PlayBtn({required this.playing, required this.onTap});
  @override
  Widget build(BuildContext context) => GestureDetector(onTap: onTap,
    child: Container(width: 48, height: 48,
      decoration: const BoxDecoration(
        color: AppColors.red, shape: BoxShape.circle),
      child: Icon(playing ? Icons.pause_rounded : Icons.play_arrow_rounded,
        color: Colors.white, size: 26)));
}

class _IconBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  const _IconBtn({required this.icon, required this.onTap});
  @override
  Widget build(BuildContext context) => GestureDetector(onTap: onTap,
    behavior: HitTestBehavior.opaque,
    child: Container(width: 40, height: 40,
      decoration: BoxDecoration(
        color: AppColors.redLight, shape: BoxShape.circle),
      child: Icon(icon, color: AppColors.red, size: 20)));
}

class _CircleIconButton extends StatelessWidget {
  final IconData icon;
  final Color? color;
  final VoidCallback onTap;
  const _CircleIconButton({required this.icon, required this.onTap, this.color});
  @override
  Widget build(BuildContext context) => GestureDetector(onTap: onTap,
    child: Container(width: 40, height: 40,
      decoration: BoxDecoration(color: Colors.white, shape: BoxShape.circle,
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.06),
              blurRadius: 8, offset: const Offset(0, 2))]),
      child: Icon(icon, size: 18, color: color ?? AppColors.textPrimary)));
}

class _PillButton extends StatelessWidget {
  final String label;
  final VoidCallback onTap;
  const _PillButton({required this.label, required this.onTap});
  @override
  Widget build(BuildContext context) => GestureDetector(onTap: onTap,
    child: Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 9),
      decoration: BoxDecoration(color: Colors.white,
          borderRadius: BorderRadius.circular(30),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.06),
              blurRadius: 8, offset: const Offset(0, 2))]),
      child: Text(label, style: const TextStyle(
        color: AppColors.red, fontWeight: FontWeight.w600, fontSize: 14))));
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();
  @override
  Widget build(BuildContext context) => Center(
    child: Column(mainAxisSize: MainAxisSize.min, children: [
      Container(width: 80, height: 80,
        decoration: BoxDecoration(color: AppColors.redLight,
            borderRadius: BorderRadius.circular(20)),
        child: const Icon(Icons.mic_none_rounded,
          color: AppColors.red, size: 40)),
      const SizedBox(height: 16),
      const Text('No recordings yet',
        style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600,
            color: AppColors.textPrimary)),
      const SizedBox(height: 6),
      const Text('Hit the mic button to record your first one.',
        style: TextStyle(fontSize: 13.5, color: AppColors.textSecondary)),
    ]));
}