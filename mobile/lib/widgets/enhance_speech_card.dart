import 'package:flutter/material.dart';
import '../main.dart' show AppColors, appCardShadow;

/// "Enhance Speech" card: a toggle + gradient strength slider controlling
/// how aggressively background noise is filtered. Meant to appear on the
/// Record/Upload result screen once a filtered recording exists, so
/// adjusting it re-processes that same recording rather than acting as a
/// silent, invisible default.
class EnhanceSpeechCard extends StatelessWidget {
  final bool enabled;
  final double strength; // 0.0-1.0
  final bool busy;
  final ValueChanged<bool> onEnabledChanged;
  final ValueChanged<double> onStrengthChanged; // live, for label/drag feedback
  final ValueChanged<double> onStrengthChangeEnd; // fires once per gesture: re-process

  const EnhanceSpeechCard({
    super.key,
    required this.enabled,
    required this.strength,
    required this.onEnabledChanged,
    required this.onStrengthChanged,
    required this.onStrengthChangeEnd,
    this.busy = false,
  });

  static const _gradient = LinearGradient(colors: [
    Color(0xFFFCA5A5), // light red
    AppColors.red,     // app accent red
    Color(0xFFB91C1C), // deep red
  ]);

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.fromLTRB(18, 16, 18, 18),
    decoration: BoxDecoration(color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: appCardShadow()),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        IgnorePointer(
          ignoring: busy,
          child: Opacity(opacity: busy ? 0.5 : 1.0,
              child: _GradientSwitch(value: enabled, onChanged: onEnabledChanged)),
        ),
        const SizedBox(width: 12),
        const Text('Enhance Speech', style: TextStyle(
            fontSize: 16, fontWeight: FontWeight.w700)),
        const Spacer(),
        if (busy)
          const SizedBox(width: 14, height: 14,
              child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.red))
        else
          Text('${(strength * 100).round()}% strength',
              style: const TextStyle(fontSize: 13.5,
                  color: AppColors.textHint, fontWeight: FontWeight.w600)),
      ]),
      if (enabled) ...[
        const SizedBox(height: 16),
        IgnorePointer(
          ignoring: busy,
          child: Opacity(opacity: busy ? 0.5 : 1.0,
            child: _GradientSlider(
              value: strength,
              gradient: _gradient,
              onChanged: onStrengthChanged,
              onChangeEnd: onStrengthChangeEnd,
            )),
        ),
      ],
    ]),
  );
}

/// Rounded pill switch, styled like the rest of the app's red accent
/// (rather than the platform's default Material switch).
class _GradientSwitch extends StatelessWidget {
  final bool value;
  final ValueChanged<bool> onChanged;
  const _GradientSwitch({required this.value, required this.onChanged});

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: () => onChanged(!value),
    behavior: HitTestBehavior.opaque,
    child: AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      width: 46, height: 26,
      padding: const EdgeInsets.all(3),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        color: value ? AppColors.red : AppColors.border,
      ),
      child: AnimatedAlign(
        duration: const Duration(milliseconds: 180),
        curve: Curves.easeOut,
        alignment: value ? Alignment.centerRight : Alignment.centerLeft,
        child: Container(width: 20, height: 20,
          decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle,
              boxShadow: [BoxShadow(color: Colors.black26, blurRadius: 3, offset: Offset(0, 1))])),
      ),
    ),
  );
}

/// A slider whose active track is a color gradient up to the current value,
/// with a flat light track beyond it. The stock Material [Slider] doesn't
/// support gradient tracks, so this layers a gradient-filled bar under a
/// fully transparent [Slider] for interaction.
class _GradientSlider extends StatelessWidget {
  final double value; // 0.0-1.0
  final Gradient gradient;
  final ValueChanged<double> onChanged;
  final ValueChanged<double> onChangeEnd;
  const _GradientSlider({
    required this.value,
    required this.gradient,
    required this.onChanged,
    required this.onChangeEnd,
  });

  @override
  Widget build(BuildContext context) {
    final v = value.clamp(0.0, 1.0);
    return SizedBox(
      height: 32,
      child: Stack(alignment: Alignment.centerLeft, children: [
        Container(height: 8,
          decoration: BoxDecoration(color: AppColors.iconBg,
              borderRadius: BorderRadius.circular(4))),
        FractionallySizedBox(
          widthFactor: v,
          child: Container(height: 8,
            decoration: BoxDecoration(gradient: gradient,
                borderRadius: BorderRadius.circular(4))),
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            trackHeight: 8,
            activeTrackColor: Colors.transparent,
            inactiveTrackColor: Colors.transparent,
            thumbColor: Colors.white,
            overlayColor: const Color(0x22EF4444), // AppColors.red @ ~13% alpha
            thumbShape: const RoundSliderThumbShape(
                enabledThumbRadius: 11, elevation: 2),
            overlayShape: const RoundSliderOverlayShape(overlayRadius: 18),
          ),
          child: Slider(
            value: v, min: 0.0, max: 1.0,
            onChanged: onChanged,
            onChangeEnd: onChangeEnd,
          ),
        ),
      ]),
    );
  }
}
