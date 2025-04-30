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
// Hide ChangeNotifierProvider from riverpod to avoid conflict with provider package
import 'package:flutter_riverpod/flutter_riverpod.dart' hide ChangeNotifierProvider; 
import 'package:provider/provider.dart'; // Import provider
import 'package:frontend/core/constants/app_strings.dart';
import 'package:frontend/core/network/websocket_service.dart'; // Import WebSocketService
import 'package:frontend/core/routes/app_routes.dart';
import 'package:frontend/core/themes/app_theme.dart';
import 'package:frontend/features/auth/application/providers/global_providers.dart';
import 'package:frontend/features/stock_ticker/application/stock_ticker_provider.dart'; // Import StockTickerProvider

void main() {
  // Instantiate the WebSocketService globally (or provide it differently if preferred)
  final webSocketService = WebSocketService();

  runApp(
    // Keep Riverpod's ProviderScope for existing features
    ProviderScope(
      // Wrap with ChangeNotifierProvider for the new feature
      child: ChangeNotifierProvider(
        create: (_) => StockTickerProvider(webSocketService),
        child: const TradingApp(),
      ),
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
