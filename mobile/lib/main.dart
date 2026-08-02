import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/home_screen.dart';
import 'screens/record_screen.dart';
import 'screens/settings_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));
  runApp(const VoiceFilterApp());
}

abstract class AppColors {
  static const Color red          = Color(0xFFEF4444);
  static const Color redLight     = Color(0xFFFEF2F2);
  static const Color green        = Color(0xFF22C55E);
  static const Color background   = Color(0xFFF4F4F4);
  static const Color textPrimary  = Color(0xFF111827);
  static const Color textSecondary= Color(0xFF6B7280);
  static const Color textHint     = Color(0xFF9CA3AF);
  static const Color border       = Color(0xFFE5E7EB);
  static const Color iconBg       = Color(0xFFF3F4F6);
}

List<BoxShadow> appCardShadow() => [
  BoxShadow(color: Colors.black.withOpacity(0.06),
            blurRadius: 14, offset: const Offset(0, 3)),
];

class VoiceFilterApp extends StatelessWidget {
  const VoiceFilterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Voice Filter AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: AppColors.background,
        colorScheme: const ColorScheme.light(
          primary: AppColors.red, onPrimary: Colors.white,
          surface: Colors.white, onSurface: AppColors.textPrimary,
        ),
        splashFactory: NoSplash.splashFactory,
        highlightColor: Colors.transparent,
      ),
      home: const MainShell(),
    );
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});
  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _idx = 0;
  static const _icons = [
    Icons.home_rounded, Icons.graphic_eq_rounded, Icons.settings_rounded
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _idx,
        children: const [HomeScreen(), RecordScreen(), SettingsScreen()],
      ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
          child: Container(
            height: 70,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(50),
              boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.10),
                  blurRadius: 24, offset: const Offset(0, 6))],
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: List.generate(3, (i) => GestureDetector(
                onTap: () => setState(() => _idx = i),
                behavior: HitTestBehavior.opaque,
                child: SizedBox(width: 72, height: 70, child: Center(
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    width: 52, height: 52,
                    decoration: BoxDecoration(
                      color: i == _idx ? AppColors.red : Colors.transparent,
                      shape: BoxShape.circle,
                    ),
                    child: Icon(_icons[i], size: 24,
                      color: i == _idx ? Colors.white : AppColors.textHint),
                  ),
                )),
              )),
            ),
          ),
        ),
      ),
    );
  }
}