import 'package:flutter/material.dart';
import '../main.dart' show AppColors, appCardShadow;
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool? _connected;

  @override
  void initState() { super.initState(); _ping(); }

  Future<void> _ping() async {
    final ok = await ApiService.ping();
    if (mounted) setState(() => _connected = ok);
  }

  @override
  Widget build(BuildContext context) {
    final pad = MediaQuery.of(context).padding.top;
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(20, pad + 24, 20, 110),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Voice Filter AI',
              style: TextStyle(fontSize: 28, fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          const Text('Hear what matters.',
              style: TextStyle(fontSize: 15, color: AppColors.textSecondary)),
          const SizedBox(height: 28),

          // Server status card
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                boxShadow: appCardShadow()),
            child: Row(children: [
              Container(width: 50, height: 50,
                decoration: BoxDecoration(
                  color: _connected == true ? AppColors.green
                       : _connected == false ? AppColors.red
                       : AppColors.iconBg,
                  borderRadius: BorderRadius.circular(12)),
                child: Icon(
                    _connected == true ? Icons.cloud_done_rounded
                  : _connected == false ? Icons.cloud_off_rounded
                  : Icons.cloud_rounded,
                  color: Colors.white)),
              const SizedBox(width: 16),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Text('Backend status',
                    style: TextStyle(fontSize: 12, color: AppColors.textHint,
                        fontWeight: FontWeight.w500)),
                const SizedBox(height: 4),
                Text(
                  _connected == null  ? 'Checking…'
                  : _connected!       ? 'Connected'
                  :                     'Unreachable — check Settings',
                  style: const TextStyle(fontSize: 14.5,
                      color: AppColors.textPrimary, fontWeight: FontWeight.w600)),
              ])),
              IconButton(onPressed: _ping,
                  icon: const Icon(Icons.refresh_rounded, color: AppColors.textHint)),
            ]),
          ),

          const SizedBox(height: 24),
          const Text('How it works',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
          const SizedBox(height: 12),

          for (final step in const [
            ('1. Record', Icons.mic_rounded,
              'Tap the mic and speak. Your laptop runs the AI in real time.'),
            ('2. Upload', Icons.upload_file_rounded,
              'Or pick an existing WAV / MP3 file from your phone.'),
            ('3. Compare', Icons.equalizer_rounded,
              'Play the cleaned version. Background noise should be gone.'),
          ])
            Padding(padding: const EdgeInsets.only(bottom: 12),
              child: Container(padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: appCardShadow()),
                child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Container(width: 36, height: 36,
                    decoration: BoxDecoration(color: AppColors.redLight,
                        borderRadius: BorderRadius.circular(10)),
                    child: Icon(step.$2, color: AppColors.red, size: 18)),
                  const SizedBox(width: 14),
                  Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(step.$1, style: const TextStyle(fontSize: 14.5,
                        fontWeight: FontWeight.w600)),
                    const SizedBox(height: 4),
                    Text(step.$3, style: const TextStyle(fontSize: 13,
                        color: AppColors.textSecondary, height: 1.45)),
                  ])),
                ]))),
        ]),
      ),
    );
  }
}