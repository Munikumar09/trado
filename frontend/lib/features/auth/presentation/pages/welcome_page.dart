/*
Documentation:
---------------
File: welcome_page.dart
Description:
  This file implements the WelcomePage which serves as the landing screen for users.
  It displays a welcome image, a greeting message, and action buttons that navigate to the sign-up or login pages.
  
Methods:
  • _buildWelcomeImage(BuildContext context):
      - Constructs the widget displaying the welcome image with error handling.
  • _buildActionButtons(BuildContext context):
      - Constructs a row containing "Sign Up" and "Login" buttons for navigation.
*/

import 'package:flutter/material.dart';
import 'package:frontend/core/constants/app_strings.dart';
import 'package:frontend/core/routes/app_routes.dart';
import 'package:frontend/features/auth/presentation/widgets/header_text_widget.dart';
import 'package:frontend/shared/buttons/primary_button.dart';
import 'package:frontend/shared/layouts/custom_background_widget.dart';

/// The welcome page of the application.
class WelcomePage extends StatelessWidget {
  const WelcomePage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context); // Get theme once for reuse

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      body: CustomBackgroundWidget(
        child: SingleChildScrollView(
          // Use ConstrainedBox for height constraint
          child: ConstrainedBox(
            constraints: BoxConstraints(
              minHeight: MediaQuery.of(context).size.height,
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24.0),
              child: Column(
                mainAxisAlignment:
                    MainAxisAlignment.center, // Center vertically
                children: [
                  const SizedBox(height: 25),
                  _buildWelcomeImage(context),
                  const SizedBox(height: 20),
                  const HeaderTextWidget(
                    //Consistent Naming
                    title: AppStrings.welcomeTitle,
                    subtitle: AppStrings.welcomeSubtitle,
                  ),
                  const SizedBox(height: 20),
                  _buildActionButtons(context),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// Builds the welcome image widget.  Extracting this makes `build` cleaner.
  Widget _buildWelcomeImage(BuildContext context) {
    return Image.asset(
      'assets/images/welcome_image.png',
      width: MediaQuery.of(context).size.width,
      height: MediaQuery.of(context).size.height / 2.5,
      fit: BoxFit.contain,
      gaplessPlayback: true,
      semanticLabel: 'Welcome illustration',
      errorBuilder: (context, error, stackTrace) {
        debugPrint('Error loading welcome image: $error');
        return const SizedBox.shrink();
      },
    );
  }

  /// Builds the row of action buttons (Sign Up and Login).
  Widget _buildActionButtons(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: PrimaryButton(
            // Use PrimaryButton
            text: AppStrings.signUp,
            onPressed: () {
              Navigator.of(context).pushNamed(AppRoutes.register);
            },
          ),
        ),
        const SizedBox(width: 20),
        Expanded(
          child: PrimaryButton(
            // Use PrimaryButton
            text: AppStrings.login,
            onPressed: () {
              Navigator.of(context).pushNamed(AppRoutes.login);
            },
          ),
        ),
      ],
    );
  }
}
