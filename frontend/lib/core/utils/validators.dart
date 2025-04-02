/*
Documentation:
---------------
Class: Validators
Description:
  Provides static methods to validate user input such as email, password, and phone number.

Methods:
  • email(value):
      - Validates that the email is not empty and correctly formatted.
      - Example: Validators.email('user@example.com') returns null if valid.
      
  • password(value):
      - Checks that the password meets length and complexity requirements.
      - Example: Validators.password('Secret123!') returns null if valid.
      
  • required(value):
      - Ensures the input field is not empty.
      
  • confirmPassword(password, confirmPassword):
      - Verifies that both password strings match.
      
  • phoneNumber(value):
      - Validates that the phone number is in the correct 10-digit format (with optional '+' prefix).
*/

// Code:
/// Provides static methods to validate user input such as email, password, and phone number.
class Validators {
  /// Validates an email string.
  static String? email(String? value) {
    if (value == null || value.isEmpty) return 'Please enter your email';
    final emailRegExp = RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$');
    if (!emailRegExp.hasMatch(value)) {
      return 'Please enter a valid email address';
    }
    return null;
  }

  /// Validates a password string for length and complexity.
  static String? password(String? value) {
    if (value == null || value.isEmpty) return 'Please enter your password';
    if (value.length < 8) return 'Password must be at least 8 characters long';
    if (!value.contains(RegExp(r'[A-Z]'))) {
      return 'Password must contain at least one uppercase letter';
    }
    if (!value.contains(RegExp(r'[0-9]'))) {
      return 'Password must contain at least one number';
    }
    if (!value.contains(RegExp(r'[!@#$%^&*(),.?":{}|<>]'))) {
      return 'Password must contain at least one special character';
    }
    return null;
  }

  /// Checks that the field is not empty.
  static String? required(String? value) {
    if (value == null || value.isEmpty) return 'This field is required';
    return null;
  }

  /// Validates that the confirmation password matches the original.
  static String? confirmPassword(String? password, String? confirmPassword) {
    if (confirmPassword == null || confirmPassword.isEmpty) {
      return 'Please confirm your password';
    }
    if (password != confirmPassword) return 'Passwords do not match';
    return null;
  }

  /// Validates a phone number in a 10-digit format .
  static String? phoneNumber(String? value) {
    if (value == null || value.isEmpty) return 'Please enter your phone number';
    final phoneRegExp = RegExp(r'^\+?\d{10}$');
    if (!phoneRegExp.hasMatch(value)) {
      return 'Please enter a valid phone number';
    }
    return null;
  }
}
