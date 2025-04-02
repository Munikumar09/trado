/*
Documentation:
---------------
File: auth_footer.dart
Description:
  This file implements the AuthFooterWidget which displays alternate authentication options.
  It shows a "Continue with" text and a social login button (e.g., using Google) for authentication.
  
Methods:
  • build(BuildContext context):
      - Constructs the widget layout containing the footer text and the social login button.
*/

// Code:
import 'package:flutter/material.dart';
import 'package:frontend/core/constants/app_strings.dart';
import 'package:frontend/core/constants/app_text_styles.dart';
import 'package:frontend/shared/buttons/icon_button.dart';

/// A widget that displays alternate authentication options.
class AuthFooterWidget extends StatelessWidget {
  const AuthFooterWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 20.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          Text(
            AppStrings.continueWith,
            style: AppTextStyles.customTextStyle(
              color: Theme.of(context).primaryColor,
              fontSize: bodyText1FontSize,
              fontWeight: bodyText1FontWeight,
            ),
          ),
          const SizedBox(height: 10),
          CustomIconButton(
            onPressed: () => {
              // implement login or sign up using Google
            },
            iconPath: "assets/images/logos/google.png",
          ),
        ],
      ),
    );
  }
}
