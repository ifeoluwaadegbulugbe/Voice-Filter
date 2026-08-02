import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

class AudioRecorderService {
  final _rec = AudioRecorder();
  String? _path;

  /// Returns true once mic permission is granted.
  Future<bool> requestPermission() async {
    if (Platform.isAndroid || Platform.isIOS) {
      final s = await Permission.microphone.request();
      debugPrint('[Recorder] mic permission: $s');
      return s.isGranted;
    }
    // Desktop — check if mic is available via the record package
    final hasPermission = await _rec.hasPermission();
    debugPrint('[Recorder] desktop hasPermission=$hasPermission');
    return hasPermission;
  }

  /// Start recording. Picks the right encoder per platform.
  /// On Windows we use the documents directory instead of temp
  /// because some systems write 0-byte files to the temp path.
  Future<String> start(String tempDir) async {
    final ts = DateTime.now().millisecondsSinceEpoch;

    final useAac  = Platform.isAndroid || Platform.isIOS;
    final ext     = useAac ? 'm4a'              : 'wav';
    final encoder = useAac ? AudioEncoder.aacLc : AudioEncoder.wav;
    final bitRate = useAac ? 128000             : 256000;

    // On Windows use documents dir — more reliable than temp for WAV writes
    String dir = tempDir;
    if (Platform.isWindows) {
      try {
        final docs = await getApplicationDocumentsDirectory();
        dir = docs.path;
        debugPrint('[Recorder] Windows: using documents dir: $dir');
      } catch (e) {
        debugPrint('[Recorder] Windows: could not get docs dir, using temp: $e');
      }
    }

    final p = '$dir/recording_$ts.$ext';

    final cfg = RecordConfig(
      encoder:       encoder,
      sampleRate:    16000,
      numChannels:   1,
      bitRate:       bitRate,
      autoGain:      useAac,
      echoCancel:    useAac,
      noiseSuppress: false,
    );

    debugPrint('[Recorder] platform=${Platform.operatingSystem} '
               'encoder=$encoder path=$p');
    await _rec.start(cfg, path: p);

    // Give Windows a moment to open the file handle
    if (Platform.isWindows) {
      await Future.delayed(const Duration(milliseconds: 300));
    }

    final isRec = await _rec.isRecording();
    debugPrint('[Recorder] isRecording=$isRec');
    if (!isRec) {
      throw Exception(
        'Recorder failed to start.\n'
        'On Windows: check Settings → System → Sound → Input\n'
        'and make sure your mic is set as the default input device.'
      );
    }

    _path = p;
    return p;
  }

  Future<String?> stop() async {
    final p = await _rec.stop();
    debugPrint('[Recorder] stopped, file=$p');

    // On Windows, flush takes a moment
    if (Platform.isWindows) {
      await Future.delayed(const Duration(milliseconds: 500));
    }

    _path = p;
    return p;
  }

  Future<bool> isRecording() => _rec.isRecording();

  String? get currentPath => _path;

  Future<Stream<Amplitude>> amplitudeStream() async {
    return _rec.onAmplitudeChanged(const Duration(milliseconds: 100));
  }

  void dispose() { _rec.dispose(); }
}