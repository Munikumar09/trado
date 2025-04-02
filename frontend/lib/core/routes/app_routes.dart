/*
Documentation:
---------------
Class: AppRoutes
Description:
  Provides a mapping of route names to their corresponding page widgets.
  Used for navigation throughout the application.

Properties:
  • initial, welcome, register, etc.:
      - Constants representing route paths.
  • pages:
      - A map of route strings to WidgetBuilders for lazy page instantiation.
      
Usage:
  Example: Navigator.pushNamed(context, AppRoutes.home);
*/

import 'package:flutter/material.dart';
import 'package:frontend/features/auth/presentation/pages/forgot_password_page.dart';
import 'package:frontend/features/auth/presentation/pages/initial_page.dart';
import 'package:frontend/features/auth/presentation/pages/login_page.dart';
import 'package:frontend/features/auth/presentation/pages/register_page.dart';
import 'package:frontend/features/auth/presentation/pages/verify_account.dart';
import 'package:frontend/features/auth/presentation/pages/welcome_page.dart';
import 'package:frontend/features/home/home.dart';

/// Provides a list of routes and their corresponding pages.
class AppRoutes {
  static const String initial = '/';
  static const String welcome = '/welcome';
  static const String register = '/register';
  static const String login = '/login';
  static const String verifyAccount = '/verify-account';
  static const String forgotPassword = '/forgot-password';
  static const String home = '/home';
  static final Map<String, WidgetBuilder> pages = {
    initial: (context) => InitialPage(),
    welcome: (context) => WelcomePage(),
    register: (context) => RegisterPage(),
    login: (context) => LoginPage(),
    forgotPassword: (context) => ForgotPasswordPage(),
    verifyAccount: (context) => VerifyAccountPage(),
    home: (context) => HomePage(),
  };
}
