// Smoke test — verifies VoiceFilterApp boots without crashing.
// Run: flutter test

import 'package:flutter_test/flutter_test.dart';
import 'package:voice_filter_ai/main.dart';

void main() {
  testWidgets('App boots without throwing', (WidgetTester tester) async {
    await tester.pumpWidget(const VoiceFilterApp());
    await tester.pump();
    expect(find.byWidgetPredicate((w) => w.runtimeType.toString() == 'MaterialApp'),
        findsOneWidget);
  });
}
