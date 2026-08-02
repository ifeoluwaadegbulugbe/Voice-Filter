import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class ApiService {
  static const String _baseUrlKey = 'server_url';
  static const String _defaultUrl = 'http://10.0.2.2:8000';
  static const String _voiceBoostKey = 'voice_boost_enabled';
  static const String _boostStrengthKey = 'boost_strength';
  static const String _noiseReductionKey = 'noise_reduction_enabled';
  static const String _noiseStrengthKey = 'noise_strength';

  static String? lastError;

  /// Get saved base URL or fallback to default
  static Future<String> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_baseUrlKey) ?? _defaultUrl;
  }

  /// Save base URL
  static Future<void> saveBaseUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_baseUrlKey, url);
  }

  /// Whether Voice Boost is enabled (default: on)
  static Future<bool> getVoiceBoostEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_voiceBoostKey) ?? true;
  }

  static Future<void> saveVoiceBoostEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_voiceBoostKey, enabled);
  }

  /// Voice Boost strength, 0.0-1.0 (default: 0.7 = 70%)
  static Future<double> getBoostStrength() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getDouble(_boostStrengthKey) ?? 0.7;
  }

  static Future<void> saveBoostStrength(double strength) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(_boostStrengthKey, strength);
  }

  /// Whether noise reduction (DeepFilterNet strength blend) is enabled (default: on)
  static Future<bool> getNoiseReductionEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_noiseReductionKey) ?? true;
  }

  static Future<void> saveNoiseReductionEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_noiseReductionKey, enabled);
  }

  /// Noise reduction strength, 0.0 (off/raw) - 1.0 (full) (default: 1.0 = full)
  static Future<double> getNoiseStrength() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getDouble(_noiseStrengthKey) ?? 1.0;
  }

  static Future<void> saveNoiseStrength(double strength) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(_noiseStrengthKey, strength);
  }

  /// Check if server is reachable
  static Future<bool> ping() async {
    lastError = null;

    try {
      final baseUrl = await getBaseUrl();
      final response = await http
          .get(Uri.parse('$baseUrl/status'))
          .timeout(const Duration(seconds: 5));

      debugPrint('[API] /status -> ${response.statusCode}');
      return response.statusCode == 200;
    } catch (e) {
      lastError = 'Ping failed: $e';
      debugPrint('[API] $lastError');
      return false;
    }
  }

  /// Upload audio file and receive filtered audio.
  ///
  /// [noiseStrength] overrides the persisted default for this one call —
  /// used by the Record/Upload result screens to re-process the same raw
  /// recording live as the user drags the "Enhance Speech" slider, without
  /// needing to have saved that value first.
  static Future<Uint8List?> filterFile(File file, {
    bool? voiceBoost,
    double? boostStrength,
    double? noiseStrength,
  }) async {
    lastError = null;

    final exists = await file.exists();
    final size = exists ? await file.length() : 0;

    debugPrint('[API] filterFile path=${file.path}, size=$size B');

    // Prevent sending empty or invalid recordings
    if (size < 4096) {
      lastError = 'Recording too small ($size B). Mic likely captured silence.';
      debugPrint('[API] $lastError');
      return null;
    }

    try {
      final baseUrl = await getBaseUrl();
      final vb = voiceBoost ?? await getVoiceBoostEnabled();
      final bs = boostStrength ?? await getBoostStrength();
      final ns = noiseStrength ?? await getNoiseStrength();
      final uri = Uri.parse('$baseUrl/filter').replace(queryParameters: {
        'voice_boost': vb.toString(),
        'boost_strength': bs.toStringAsFixed(2),
        'noise_strength': ns.toStringAsFixed(2),
      });

      debugPrint('[API] POST $uri');

      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('audio', file.path));

      final response = await request.send().timeout(const Duration(seconds: 60));

      debugPrint(
        '[API] status=${response.statusCode}, content-length=${response.contentLength}',
      );

      if (response.statusCode != 200) {
        final body = await response.stream.bytesToString();
        lastError =
            'HTTP ${response.statusCode}: ${body.substring(0, body.length.clamp(0, 200))}';
        debugPrint('[API] $lastError');
        return null;
      }

      final bytes = await response.stream.toBytes();
      debugPrint('[API] received ${bytes.length} B');

      if (bytes.isEmpty) {
        lastError = 'Server returned empty audio.';
        return null;
      }

      return bytes;
    } catch (e) {
      lastError = 'Exception: $e';
      debugPrint('[API] $lastError');
      return null;
    }
  }
}