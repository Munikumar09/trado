import 'package:flutter/material.dart';
import 'package:frontend/core/constants/app_strings.dart';
import 'package:frontend/core/constants/app_text_styles.dart';
import 'package:frontend/core/routes/app_routes.dart';
import 'package:frontend/features/auth/presentation/widgets/header_text_widget.dart';
import 'package:frontend/features/auth/presentation/widgets/auth_footer.dart';
import 'package:frontend/shared/buttons/custom_button.dart';
import 'package:frontend/shared/inputs/custom_text_field.dart';
import 'package:frontend/shared/layouts/custom_background_widget.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      body: CustomBackgroundWidget(
        child: SingleChildScrollView(
          child: Container(
            height: MediaQuery.of(context).size.height,
            padding: const EdgeInsets.symmetric(horizontal: 24.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Spacer(),
                HeaderTextWidget(
                  title: AppStrings.loginTitle,
                  subtitle: AppStrings.loginSubtitle,
                ),
                Spacer(),
                Form(
                  key: _formKey,
                  child: Column(
                    children: [
                      CustomTextField(
                        hintText: AppStrings.email,
                        labelText: AppStrings.email,
                        keyboardType: TextInputType.emailAddress,
                        controller: _emailController,
                      ),
                      const SizedBox(height: 16),
                      CustomTextField(
                        hintText: AppStrings.password,
                        isPassword: true,
                        labelText: AppStrings.password,
                        controller: _passwordController,
                        autocorrect: false,
                        enableSuggestions: false,
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 10),
                Align(
                  alignment: Alignment.centerRight,
                  child: TextButton(
                    onPressed: () {
                      Navigator.of(context).pushNamed(AppRoutes.forgotPassword);
                    },
                    child: Text(
                      AppStrings.forgotPassword,
                      style: AppTextStyles.customTextStyle(
                          color: theme.colorScheme.secondary,
                          fontSize: heading3FontSize,
                          fontWeight: heading3FontWeight),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                CustomButton(
                  text: AppStrings.login,
                  onPressed: () {
                    if (_formKey.currentState?.validate() ?? false) {
                      // Handle login
                    }
                  },
                ),
                const SizedBox(height: 10),
                TextButton(
                  onPressed: () {
                    Navigator.of(context)
                        .pushReplacementNamed(AppRoutes.register);
                  },
                  child: Text(
                    AppStrings.createAccount,
                    style: AppTextStyles.customTextStyle(
                        color: const Color(0xFF494949),
                        fontSize: heading3FontSize,
                        fontWeight: heading3FontWeight),
                  ),
                ),
                Spacer(),
                const AuthFooterWidget(),
                Spacer(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
