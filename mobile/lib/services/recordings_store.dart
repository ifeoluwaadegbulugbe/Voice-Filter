import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class Recording {
  final String id;
  final String name;
  final String enhancedPath;
  final String? originalPath;
  final DateTime createdAt;
  final double durationSec;

  Recording({
    required this.id,
    required this.name,
    required this.enhancedPath,
    this.originalPath,
    required this.createdAt,
    required this.durationSec,
  });

  Recording copyWith({String? name}) => Recording(
    id: id,
    name: name ?? this.name,
    enhancedPath: enhancedPath,
    originalPath: originalPath,
    createdAt: createdAt,
    durationSec: durationSec,
  );

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'enhancedPath': enhancedPath,
    'originalPath': originalPath,
    'createdAt': createdAt.toIso8601String(),
    'durationSec': durationSec,
  };

  static Recording fromJson(Map<String, dynamic> j) => Recording(
    id:           j['id'],
    name:         j['name'],
    enhancedPath: j['enhancedPath'],
    originalPath: j['originalPath'],
    createdAt:    DateTime.parse(j['createdAt']),
    durationSec:  (j['durationSec'] as num).toDouble(),
  );
}

class RecordingsStore {
  static const _kKey = 'recordings_list_v1';

  static Future<List<Recording>> loadAll() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_kKey);
    if (raw == null) return [];
    try {
      final List<dynamic> arr = jsonDecode(raw);
      return arr
          .map((e) => Recording.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  static Future<void> add(Recording r) async {
    final list = await loadAll();
    list.insert(0, r);
    await _save(list);
  }

  static Future<void> remove(String id) async {
    final list = await loadAll();
    list.removeWhere((r) => r.id == id);
    await _save(list);
  }

  static Future<void> rename(String id, String newName) async {
    final list = await loadAll();
    final idx = list.indexWhere((r) => r.id == id);
    if (idx >= 0) {
      list[idx] = list[idx].copyWith(name: newName);
      await _save(list);
    }
  }

  static Future<void> _save(List<Recording> list) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _kKey,
      jsonEncode(list.map((r) => r.toJson()).toList()),
    );
  }
}