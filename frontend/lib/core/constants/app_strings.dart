/*
Documentation:
---------------
Class: AppStrings
Description:
  Contains constant string values used throughout the application for UI text, messages, and labels.
  
Examples:
  • AppStrings.appName: Used to display the name of the application.
  • AppStrings.welcomeTitle: Shown on the welcome screen.
  • AppStrings.login, AppStrings.signUp, etc.: Utilized in authentication screens.
*/

// Code:
/// Contains constant string values used throughout the application for UI text, messages, and labels.
class AppStrings {
  // App Name
  static const String appName = 'Paper Trading App';

  // Welcome Screen Strings
  static const String welcomeTitle = 'Master the Market\nwith Zero Risk';
  static const String welcomeSubtitle =
      'Trade stocks and options virtually.\nLearn, practice, and grow risk-free';
  static const String signUp = 'Sign Up';
  static const String login = 'Login';
  static const String haveAccount = 'Already have an account?';

  // Login Screen Strings
  static const String loginTitle = 'Login here';
  static const String loginSubtitle = "Welcome back you've been missed!";
  static const String loginSuccessMessage = 'Login successful!';
  static const String forgotPassword = 'Forgot your password?';

  // Registration Screen Strings
  static const String createAccount = 'Create an account';
  static const String registerTitle = 'Create Account';
  static const String registerSubtitle =
      'Create an account to start your\nrisk-free trading journey today!';

  // Reset Password Screen Strings
  static const String resetPassword = 'Reset Your Password';
  static const String resetPasswordSubtitle =
      'Enter your email address and we\'ll send you an otp to reset your password.';
  static const String sendOtp = 'Send OTP';
  static const String newPassword = 'New Password';
  static const String confirmNewPassword = 'Confirm New Password';
  static const String send = 'Send';

  // Verification Screen Strings
  static const String verifyAccount = 'Verify Your Account';
  static const String verifyAccountSubtitle =
      'Enter the OTP sent to your email address to verify your account.';
  static const String emailOtp = 'Enter Email OTP';
  static const String phoneOtp = 'Enter Phone OTP';
  static const String verify = 'Verify';

  // Common Field Strings
  static const String email = 'Email';
  static const String password = 'Password';
  static const String confirmPassword = 'Confirm password';
  static const String username = 'Username';
  static const String phoneNumber = 'Phone Number';
  static const String dateOfBirth = 'Date of Birth';
  static const String gender = 'Gender';

  // Error Messages
  static const String userNotVerified = 'not verified';
  static const String genericError = 'An error occurred. Please try again';

  // Social Login Strings
  static const String continueWith = 'Or continue with';
}
