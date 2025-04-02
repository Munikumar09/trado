/*
Documentation:
---------------
File: main.dart
Description:
  This is the entry point of the application. It initializes the provider scope and sets up the TradingApp widget.
  The TradingApp class configures the MaterialApp with themes, routes, and navigator key from Riverpod.
  It also provides gesture detection to dismiss the keyboard and includes route observers for navigation monitoring.

Methods:
  • main():
      - Runs the app within a ProviderScope.
  • TradingApp.build(BuildContext context, WidgetRef ref):
      - Builds the MaterialApp with app configuration such as themes, routes, navigator key, and gesture handling.
*/

// Code:
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/core/constants/app_strings.dart';
import 'package:frontend/core/routes/app_routes.dart';
import 'package:frontend/core/themes/app_theme.dart';
import 'package:frontend/features/auth/application/providers/global_providers.dart';

void main() {
  runApp(
    const ProviderScope(
      child: TradingApp(),
    ),
  );
}

/// The main application widget that configures the MaterialApp.
class TradingApp extends ConsumerWidget {
  const TradingApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final navigatorKey = ref.read(navigatorKeyProvider);

    return MaterialApp(
      title: AppStrings.appName,
      theme: AppThemes.lightTheme,
      themeMode: ThemeMode.system,
      navigatorKey: navigatorKey,
      initialRoute: AppRoutes.initial,
      routes: AppRoutes.pages,
      builder: (context, child) {
        return GestureDetector(
          onTap: () => FocusManager.instance.primaryFocus?.unfocus(),
          child: child,
        );
      },
      navigatorObservers: [HeroController()],
    );
  }
}
