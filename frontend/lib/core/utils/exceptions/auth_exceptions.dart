/*
Documentation:
---------------
Module: Authentication Exceptions
Description:
  Defines custom exceptions related to authentication failures, such as signup or login issues,
  token problems, and email verification failures. These exceptions extend from AppException.
  
Classes:
  • AuthException: Base exception for authentication errors.
  • SignupFailedException: Thrown when user signup fails.
  • LoginFailedException: Thrown when login fails.
  • EmailNotVerifiedException: Thrown when email verification is required.
  • VerificationFailedException: Thrown when the account verification process fails.
  • TokenRefreshFailedException: Thrown when refreshing the auth token fails.
  • NoRefreshTokenException: Thrown when no refresh token is available.
  • LogoutFailedException: Thrown when logout fails.
  • TokenStorageException: Thrown when there is a token storage issue.
  • PasswordResetFailedException: Thrown when password reset fails.
  • RefreshTokenExpiredException: Thrown when the refresh token has expired.
  • SendVerificationCodeFailedException: Thrown when sending the verification code to email fails.
*/

import 'package:frontend/core/utils/exceptions/app_exceptions.dart';

/// Base exception for authentication errors.
class AuthException extends AppException {
  AuthException(super.message, [super.stackTrace]);
}

/// Thrown when user signup fails.
class SignupFailedException extends AuthException {
  SignupFailedException(super.message, [super.stackTrace]);
}

/// Thrown when login fails.
class LoginFailedException extends AuthException {
  LoginFailedException(super.message, [super.stackTrace]);
}

/// Thrown when email verification is required.
class EmailNotVerifiedException extends AuthException {
  EmailNotVerifiedException(super.message, [super.stackTrace]);
}

/// Thrown when the account verification process fails.
class VerificationFailedException extends AuthException {
  VerificationFailedException(super.message, [super.stackTrace]);
}

/// Thrown when refreshing the authentication token fails.
class TokenRefreshFailedException extends AuthException {
  TokenRefreshFailedException(super.message, [super.stackTrace]);
}

/// Thrown when no refresh token is available.
class NoRefreshTokenException extends AuthException {
  NoRefreshTokenException(super.message, [super.stackTrace]);
}

/// Thrown when logout fails.
class LogoutFailedException extends AuthException {
  LogoutFailedException(super.message, [super.stackTrace]);
}

/// Thrown when there is a token storage issue.
class TokenStorageException extends AuthException {
  TokenStorageException(super.message);
}

/// Thrown when password reset fails.
class PasswordResetFailedException extends AuthException {
  PasswordResetFailedException(super.message, [super.stackTrace]);
}

/// Thrown when the refresh token has expired.
class RefreshTokenExpiredException extends AuthException {
  RefreshTokenExpiredException(super.message, [super.stackTrace]);
}

/// Thrown when sending the verification code to email fails.
class SendVerificationCodeFailedException extends AuthException {
  SendVerificationCodeFailedException(super.message, [super.stackTrace]);
}
