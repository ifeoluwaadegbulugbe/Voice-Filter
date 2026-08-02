import 'package:flutter/material.dart';
import '../main.dart' show AppColors, appCardShadow;
import '../services/api_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String _url = 'http://10.0.2.2:8000';
  bool? _ok;
  bool _voiceBoost = true;
  double _boostStrength = 0.7;

  @override
  void initState() { super.initState(); _load(); }

  Future<void> _load() async {
    final url = await ApiService.getBaseUrl();
    final ok  = await ApiService.ping();
    final voiceBoost = await ApiService.getVoiceBoostEnabled();
    final boostStrength = await ApiService.getBoostStrength();
    if (mounted) setState(() {
      _url = url; _ok = ok;
      _voiceBoost = voiceBoost; _boostStrength = boostStrength;
    });
  }

  Future<void> _setVoiceBoost(bool enabled) async {
    setState(() => _voiceBoost = enabled);
    await ApiService.saveVoiceBoostEnabled(enabled);
  }

  Future<void> _setBoostStrength(double strength) async {
    setState(() => _boostStrength = strength);
    await ApiService.saveBoostStrength(strength);
  }

  Future<void> _edit() async {
    final ctl = TextEditingController(text: _url);
    final result = await showDialog<String>(context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('Server URL',
            style: TextStyle(fontWeight: FontWeight.w700)),
        content: TextField(controller: ctl, autofocus: true,
            decoration: const InputDecoration(
                hintText: 'http://192.168.x.x:8000',
                helperText: 'Use http://10.0.2.2:8000 from the Android emulator')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel')),
          ElevatedButton(
              onPressed: () => Navigator.pop(ctx, ctl.text.trim()),
              style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.red, foregroundColor: Colors.white),
              child: const Text('Save')),
        ]));
    if (result != null && result.isNotEmpty) {
      var u = result;
      if (!u.startsWith('http')) u = 'http://$u';
      await ApiService.saveBaseUrl(u);
      _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.of(context).padding.top;
    return Scaffold(
      backgroundColor: AppColors.background,
      body: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        // Hero header
        Container(width: double.infinity,
          padding: EdgeInsets.fromLTRB(24, pad + 20, 24, 30),
          decoration: const BoxDecoration(color: AppColors.red),
          child: const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Hear what matters', style: TextStyle(color: Colors.white,
                fontSize: 22, fontWeight: FontWeight.w700)),
            SizedBox(height: 8),
            Text('Customize how your AI filters speech from background noise.',
                style: TextStyle(color: Colors.white70, fontSize: 14, height: 1.5)),
          ])),

        Expanded(child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 22, 20, 110),
          children: [
            const Text('General Settings',
                style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
            const SizedBox(height: 14),

            _Tile(icon: Icons.language_rounded, title: 'Server URL',
                trailing: _url, onTap: _edit),
            const SizedBox(height: 10),

            _Tile(
              icon: _ok == true  ? Icons.cloud_done_rounded
                 : _ok == false  ? Icons.cloud_off_rounded
                 :                 Icons.cloud_rounded,
              title: 'Status',
              trailing: _ok == null ? '…' : (_ok! ? 'Connected' : 'Unreachable'),
              hasChevron: false,
              onTap: _load),

            const SizedBox(height: 26),
            const Text('Voice Enhancement',
                style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
            const SizedBox(height: 14),

            _VoiceBoostCard(
              enabled: _voiceBoost,
              strength: _boostStrength,
              onEnabledChanged: _setVoiceBoost,
              onStrengthChanged: _setBoostStrength,
            ),

            const SizedBox(height: 26),
            const Text('About',
                style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
            const SizedBox(height: 14),

            _Tile(icon: Icons.phone_android_rounded, title: 'App Version',
                trailing: 'v1.0.0', hasChevron: false),
            const SizedBox(height: 10),

            _Tile(icon: Icons.info_outline_rounded, title: 'About',
              onTap: () => showAboutDialog(context: context,
                  applicationName: 'Voice Filter AI',
                  applicationVersion: '1.0.0',
                  children: const [
                    Text('AI-powered speech enhancement for hearing-aid '
                         'applications. Powered by DeepFilterNet 3.'),
                  ])),
          ])),
      ]),
    );
  }
}

class _VoiceBoostCard extends StatelessWidget {
  final bool enabled;
  final double strength;
  final ValueChanged<bool> onEnabledChanged;
  final ValueChanged<double> onStrengthChanged;
  const _VoiceBoostCard({
    required this.enabled,
    required this.strength,
    required this.onEnabledChanged,
    required this.onStrengthChanged,
  });

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
    decoration: BoxDecoration(color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: appCardShadow()),
    child: Column(children: [
      Row(children: [
        Container(width: 36, height: 36,
          decoration: BoxDecoration(color: AppColors.iconBg,
              borderRadius: BorderRadius.circular(10)),
          child: const Icon(Icons.volume_up_rounded, size: 18,
              color: AppColors.textSecondary)),
        const SizedBox(width: 14),
        const Expanded(child: Text('Voice Boost',
            style: TextStyle(fontSize: 15, fontWeight: FontWeight.w500))),
        Switch(
          value: enabled,
          onChanged: onEnabledChanged,
          activeThumbColor: AppColors.red,
        ),
      ]),
      if (enabled) ...[
        const Divider(height: 1),
        Padding(
          padding: const EdgeInsets.only(top: 4, bottom: 10),
          child: Row(children: [
            const Text('Strength', style: TextStyle(
                fontSize: 13.5, color: AppColors.textHint)),
            Expanded(child: Slider(
              value: strength,
              min: 0.0,
              max: 1.0,
              divisions: 20,
              activeColor: AppColors.red,
              label: '${(strength * 100).round()}%',
              onChanged: onStrengthChanged,
            )),
            SizedBox(width: 38, child: Text('${(strength * 100).round()}%',
                textAlign: TextAlign.right,
                style: const TextStyle(fontSize: 13.5,
                    color: AppColors.textHint, fontWeight: FontWeight.w600))),
          ]),
        ),
      ],
    ]),
  );
}

class _Tile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? trailing;
  final bool hasChevron;
  final VoidCallback? onTap;
  const _Tile({required this.icon, required this.title,
      this.trailing, this.hasChevron = true, this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap, behavior: HitTestBehavior.opaque,
    child: Container(padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
      decoration: BoxDecoration(color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: appCardShadow()),
      child: Row(children: [
        Container(width: 36, height: 36,
          decoration: BoxDecoration(color: AppColors.iconBg,
              borderRadius: BorderRadius.circular(10)),
          child: Icon(icon, size: 18, color: AppColors.textSecondary)),
        const SizedBox(width: 14),
        Expanded(child: Text(title,
            style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w500))),
        if (trailing != null) ...[
          ConstrainedBox(constraints: const BoxConstraints(maxWidth: 180),
            child: Text(trailing!, textAlign: TextAlign.right, maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(fontSize: 13.5, color: AppColors.textHint))),
          const SizedBox(width: 6),
        ],
        if (hasChevron) const Icon(Icons.chevron_right_rounded,
            color: AppColors.border, size: 22),
      ]),
    ));
}