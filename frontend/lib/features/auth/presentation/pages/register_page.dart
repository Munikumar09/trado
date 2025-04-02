/*
Documentation:
---------------
File: register_page.dart
Description:
  This file implements the RegisterPage which provides a sign-up form for new users.
  It includes input fields such as username, email, password, confirm password, phone number, date of birth, and gender.
  The file handles form validation, opens a date picker for the date field, and dispatches a signup request via the auth provider.
  
Methods:
  • _selectDate(BuildContext context):
      - Opens a date picker and updates the 'date' controller with the chosen date.
  • _onSignUpPressed():
      - Validates the form, creates a signup request, and calls the signup provider.
  • _buildForm():
      - Constructs the registration form widget.
  • _buildFormField({…}):
      - Builds a reusable form field for text inputs.
  • _buildDateField():
      - Constructs a date input field integrated with a date picker.
  • _buildGenderDropdown():
      - Builds a dropdown for selecting gender.
  • _buildSubmitButton():
      - Builds the submit button with a loading indicator when processing.
  • _buildLoginLink():
      - Provides a link to navigate to the login screen.
*/

// Code:
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/core/constants/app_strings.dart';
import 'package:frontend/core/constants/app_text_styles.dart';
import 'package:frontend/core/routes/app_routes.dart';
import 'package:frontend/core/utils/validators.dart';
import 'package:frontend/features/auth/application/model/signup_request.dart';
import 'package:frontend/features/auth/application/providers/auth_providers.dart';
import 'package:frontend/features/auth/application/state/auth_state.dart';
import 'package:frontend/features/auth/presentation/widgets/auth_footer.dart';
import 'package:frontend/features/auth/presentation/widgets/header_text_widget.dart';
import 'package:frontend/shared/buttons/primary_button.dart'; // Import PrimaryButton
import 'package:frontend/shared/helpers/custom_snackbar.dart';
import 'package:frontend/shared/inputs/custom_drop_down_box.dart';
import 'package:frontend/shared/inputs/custom_text_field.dart';
import 'package:frontend/shared/layouts/custom_background_widget.dart';
import 'package:frontend/shared/loaders/loading_indicator.dart';

class RegisterPage extends ConsumerStatefulWidget {
  const RegisterPage({super.key});

  @override
  ConsumerState<RegisterPage> createState() => _RegisterPageState();
}

/// State for [RegisterPage].
///
/// Manages the form, input fields, and registration logic.
class _RegisterPageState extends ConsumerState<RegisterPage> {
  final _formKey = GlobalKey<FormState>();

  // Use a Map for controllers, improves organization and disposal
  final Map<String, TextEditingController> _controllers = {
    'date': TextEditingController(),
    'password': TextEditingController(),
    'phone': TextEditingController(),
    'username': TextEditingController(),
    'email': TextEditingController(),
    'confirmPassword': TextEditingController(),
  };
  String? _selectedGender; // Store the selected gender

  @override
  void dispose() {
    // Dispose all controllers in the map
    for (final controller in _controllers.values) {
      controller.dispose();
    }
    super.dispose();
  }

  /// Opens a date picker and updates the 'date' controller.
  Future<void> _selectDate(BuildContext context) async {
    final pickedDate = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
    );
    if (pickedDate != null) {
      // Format the date as needed.  Using a consistent format is important.
      _controllers['date']!.text =
          "${pickedDate.day}/${pickedDate.month}/${pickedDate.year}";
    }
  }

  /// Handles form submission.  Validates, creates a request, and calls the signup provider.
  Future<void> _onSignUpPressed() async {
    if (_formKey.currentState!.validate()) {
      // All validation checks have passed at this point
      final request = SignupRequest(
        username: _controllers['username']!.text,
        email: _controllers['email']!.text,
        password: _controllers['password']!.text,
        confirmPassword: _controllers['confirmPassword']!.text,
        phoneNumber: _controllers['phone']!.text,
        dateOfBirth: _controllers['date']!.text,
        gender: _selectedGender!,
      );

      await ref.read(authNotifierProvider.notifier).signup(request);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    // Listen for auth state changes to handle navigation and errors.
    ref.listen<AuthState>(authNotifierProvider, (previous, next) {
      if (previous?.status == AuthStatus.loading &&
          next.status != AuthStatus.loading) {
        if (next.status == AuthStatus.error && next.error != null) {
          context.showErrorSnackBar(next.error!);
        } else if (next.status == AuthStatus.verificationPending) {
          context.showSuccessSnackBar(
              'Account created! Please verify your email.');
          Navigator.pushNamed(
            context,
            AppRoutes.verifyAccount,
            arguments: {'email': next.email}, // Pass email for verification
          );
        }
      }
    });

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      body: CustomBackgroundWidget(
        child: SingleChildScrollView(
          child: ConstrainedBox(
            constraints: BoxConstraints(
              minHeight: MediaQuery.of(context).size.height,
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(height: 16),
                  const HeaderTextWidget(
                    // Consistent naming
                    title: AppStrings.registerTitle,
                    subtitle: AppStrings.registerSubtitle,
                  ),
                  const SizedBox(height: 16),
                  _buildForm(), // Build the form
                  const SizedBox(height: 16),
                  const AuthFooterWidget(), // Consistent naming
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// Builds the registration form widget.
  Widget _buildForm() {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          _buildFormField(
            controller: _controllers['username']!,
            label: AppStrings.username,
          ),
          _buildFormField(
            controller: _controllers['email']!,
            label: AppStrings.email,
            keyboardType: TextInputType.emailAddress,
            validator: Validators.email,
          ),
          _buildFormField(
            controller: _controllers['password']!,
            label: AppStrings.password,
            isPassword: true,
            validator: Validators.password,
          ),
          _buildFormField(
            controller: _controllers['confirmPassword']!,
            label: AppStrings.confirmPassword,
            isPassword: true,
            validator: (value) => Validators.confirmPassword(
              _controllers['password']!
                  .text, // Pass the password for comparison
              value,
            ),
          ),
          _buildFormField(
            controller: _controllers['phone']!,
            label: AppStrings.phoneNumber,
            keyboardType: TextInputType.phone,
            validator: Validators.phoneNumber,
          ),
          _buildDateField(), // Use the dedicated date field builder
          const SizedBox(height: 10),
          _buildGenderDropdown(), // Use dedicated gender dropdown builder
          const SizedBox(height: 20),
          _buildSubmitButton(), // Dedicated submit button builder
          _buildLoginLink(), // Dedicated login link builder
        ],
      ),
    );
  }

  /// Builds a reusable form field widget.
  Widget _buildFormField({
    required TextEditingController controller,
    required String label,
    TextInputType? keyboardType,
    bool isPassword = false,
    String? Function(String?)? validator,
  }) {
    return Column(
      children: [
        CustomTextField(
          controller: controller,
          labelText: label,
          hintText: label,
          keyboardType: keyboardType,
          isPassword: isPassword,
          autocorrect: !isPassword,
          enableSuggestions: !isPassword,
          validator: validator,
        ),
        const SizedBox(height: 10),
      ],
    );
  }

  /// Builds the date of birth field with a date picker.
  Widget _buildDateField() {
    return CustomTextField(
      readOnly: true, // Prevent manual text input
      labelText: AppStrings.dateOfBirth,
      hintText: AppStrings.dateOfBirth,
      suffixIcon: Icons.calendar_today, // Calendar icon
      controller: _controllers['date']!, // Use the date controller
      onSuffixTap: () => _selectDate(context), // Open date picker on tap
    );
  }

  /// Builds the gender dropdown field.
  Widget _buildGenderDropdown() {
    return CustomDropdown(
      labelText: AppStrings.gender,
      options: const ["Male", "Female", "Other"], // Gender options
      onChanged: (value) =>
          setState(() => _selectedGender = value), // Update selected gender
    );
  }

  /// Builds the submit button with loading state handling.
  Widget _buildSubmitButton() {
    final authState = ref.watch(authNotifierProvider);

    return PrimaryButton(
      text: AppStrings.signUp, // Always show "Sign Up"
      onPressed: authState.status == AuthStatus.loading
          ? null
          : _onSignUpPressed, // Disable when loading
      child: authState.status == AuthStatus.loading
          ? LoadingIndicator()
          : null, // Show indicator when loading
    );
  }

  /// Builds the "Already have an account?" link.
  Widget _buildLoginLink() {
    return TextButton(
      onPressed: () => Navigator.pushReplacementNamed(context, AppRoutes.login),
      child: Text(
        AppStrings.haveAccount,
        style: AppTextStyles.customTextStyle(
            color: Color(0xFF494949),
            fontSize: heading3FontSize,
            fontWeight: FontWeight.w500),
      ),
    );
  }
}
