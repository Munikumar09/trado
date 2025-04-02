/*
Documentation:
---------------
Class: ApiEndpoints
Description:
  Contains constant endpoint paths for the API.
  Defines the base URL and all authentication and protected endpoint paths.
  
Examples:
  • ApiEndpoints.baseUrl: Base URL of the API.
  • ApiEndpoints.signin: Endpoint for user sign-in.
  • ApiEndpoints.refreshToken: Endpoint to refresh tokens.
*/

// Code:
/// Contains constant endpoint paths for the API.
class ApiEndpoints {
  static const String baseUrl = "http://10.0.2.2:8000";

  static const String _authPrefix = '/authentication';
  static const String sendVerification = '$_authPrefix/send-verification-code';
  static const String verifyCode = '$_authPrefix/verify-code';
  static const String refreshToken = '$_authPrefix/refresh-token';
  static const String signin = '$_authPrefix/signin';
  static const String logout = '$_authPrefix/logout';
  static const String signup = '$_authPrefix/signup';
  static const String protected = '$_authPrefix/protected-endpoint';
  static const String sendResetPasswordCode =
      '$_authPrefix/send-reset-password-code';
  static const String resetPassword = '$_authPrefix/reset-password';
}
