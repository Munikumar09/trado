/*
Documentation:
---------------
Class: AuthNotifier
Description:
  Manages authentication state and operations. Uses AuthRepository for API calls and TokenStorage for managing tokens.
  
Methods:
  • signin(email, password):
      - Signs in the user and updates auth state.
      - Example: await authNotifier.signin('user@example.com', 'password123');
      
  • signup(request):
      - Registers a new user.
      
  • sendVerificationCode(email):
      - Sends a code for email verification.
      
  • verifyVerificationCode(email, code):
      - Verifies the provided code.
      
  • sendResetPasswordCode(email):
      - Sends a reset password code.
      
  • resetPassword(email, code, newPassword):
      - Resets the user's password.
      
  • logout():
      - Logs out the user by clearing tokens.
      
  • checkAuthState():
      - Checks current auth state.
      
  • _executeAuthOperation(...):
      - Helper for executing auth operations with error handling.
*/

// Code:
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/core/utils/exceptions/app_exceptions.dart';
import 'package:frontend/core/utils/exceptions/auth_exceptions.dart';
import 'package:frontend/features/auth/application/model/signup_request.dart';
import 'package:frontend/features/auth/application/repository/auth_repository.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';
import 'package:frontend/features/auth/application/state/auth_state.dart';
import 'package:logger/logger.dart';

/// Manages authentication state and operations using [AuthRepository] and [TokenStorage].
class AuthNotifier extends StateNotifier<AuthState> {
  final Logger _logger = Logger();
  final AuthRepository _authRepository;
  final TokenStorage _tokenStorage;

  /// Constructs an instance of [AuthNotifier] with the provided repository and token storage.
  AuthNotifier(this._authRepository, this._tokenStorage)
      : super(AuthState.initial()) {
    _logger.d("AuthNotifier initialized");
  }

  /// Signs in a user with their [email] and [password].
  Future<void> signin(String email, String password) async {
    _logger.i("Attempting to sign in with email: $email");
    await _executeAuthOperation(
      operation: () async {
        await _authRepository.signin(email, password);
        final tokens = await _tokenStorage.getTokens();
        _logger.i("Sign in successful. Access token: ${tokens.accessToken}");
        return AuthState.authenticated(tokens.accessToken!);
      },
      specificErrorHandler: (error) {
        if (error is EmailNotVerifiedException) {
          _logger.w("Email not verified: $email");
          return AuthState.unverified(
            email: email,
            error: null,
          );
        }
        _logger.e("Sign in failed: $error");
        return null;
      },
    );
  }

  /// Signs up a new user with the provided [request] details.
  Future<void> signup(SignupRequest request) async {
    _logger.i("Attempting to sign up with email: ${request.email}");
    await _executeAuthOperation(
      operation: () async {
        await _authRepository.signup(request);
        _logger.i("Sign up successful for email: ${request.email}");
        return state.copyWith(
          status: AuthStatus.verificationPending,
          email: request.email,
        );
      },
      specificErrorHandler: (error) {
        _logger.e("Sign up failed: $error");
        return null;
      },
    );
  }

  /// Sends a verification code to the specified [email].
  Future<void> sendVerificationCode(String email) async {
    _logger.i("Sending verification code to email: $email");
    await _executeAuthOperation(
      operation: () async {
        await _authRepository.sendVerificationCode(email);
        _logger.i("Verification code sent to email: $email");
        return state.copyWith(
          status: AuthStatus.verificationSent,
          email: email,
        );
      },
      specificErrorHandler: (error) {
        _logger.e("Sending verification code failed: $error");
        return null;
      },
    );
  }

  /// Verifies the [code] sent to the [email] and updates state accordingly.
  Future<void> verifyVerificationCode(String email, String code) async {
    _logger.i("Verifying code: $code for email: $email");
    await _executeAuthOperation(
      operation: () async {
        await _authRepository.verifyVerificationCode(email, code);
        _logger.i("Verification successful for email: $email");
        return AuthState.initial();
      },
      specificErrorHandler: (error) {
        _logger.e("Verification code failed: $error");
        return null;
      },
    );
  }

  /// Sends a reset password code to the given [email].
  Future<void> sendResetPasswordCode(String email) async {
    _logger.i("Sending reset password code to email: $email");
    await _executeAuthOperation(
      operation: () async {
        await _authRepository.sendResetPasswordCode(email);
        _logger.i("Reset password code sent to email: $email");
        return state.copyWith(
          status: AuthStatus.verificationSent,
          email: email,
        );
      },
      specificErrorHandler: (error) {
        _logger.e("Sending reset password code failed: $error");
        return null;
      },
    );
  }

  /// Resets the user's password using the provided [email], [code], and [newPassword].
  Future<void> resetPassword(
      String email, String code, String newPassword) async {
    _logger.i("Resetting password for email: $email");
    await _executeAuthOperation(
      operation: () async {
        await _authRepository.resetPassword(email, code, newPassword);
        _logger.i("Password reset successful for email: $email");
        return AuthState.initial();
      },
      specificErrorHandler: (error) {
        _logger.e("Password reset failed: $error");
        return null;
      },
    );
  }

  /// Logs the user out by clearing authentication credentials.
  Future<void> logout() async {
    _logger.i("Logging out");
    await _executeAuthOperation(
      operation: () async {
        await _authRepository.logout();
        _logger.i("Logout successful");
        return AuthState.unauthenticated();
      },
      allowLoading: false,
      specificErrorHandler: (error) {
        _logger.e("Logout failed: $error");
        return null;
      },
    );
  }

  /// Checks the current authentication state using stored tokens.
  Future<void> checkAuthState() async {
    _logger.i("Checking authentication state");
    await _executeAuthOperation(
      operation: () async {
        final tokens = await _tokenStorage.getTokens();
        if (tokens.accessToken == null) {
          _logger.i("No access token found");
          return AuthState.unauthenticated();
        }
        await _authRepository.checkAuthState();
        final validTokens = await _tokenStorage.getTokens();
        if (validTokens.accessToken == null) {
          _logger.i("No access token found after checkAuthState");
          return AuthState.unauthenticated();
        }
        _logger.i(
            "User is authenticated. Access Token: ${validTokens.accessToken}");
        return AuthState.authenticated(validTokens.accessToken!);
      },
      allowLoading: false,
    );
  }

  /// Executes an authentication operation with built-in error handling.
  ///
  /// [operation] is the asynchronous function performing the authentication work.
  /// [specificErrorHandler] allows custom handling for specific [AppException] errors.
  /// [allowLoading] indicates whether loading state should be set before execution.
  Future<void> _executeAuthOperation({
    required Future<AuthState> Function() operation,
    AuthState? Function(AppException error)? specificErrorHandler,
    bool allowLoading = true,
  }) async {
    try {
      if (allowLoading) {
        _logger.d("Setting AuthStatus to loading");
        state = state.copyWith(status: AuthStatus.loading);
      }

      final newState = await operation();
      // Schedule the state update immediately after the successful operation.
      Future.microtask(() {
        _logger.d("Auth operation successful, updating state");
        state = newState;
      });
    } on AppException catch (error) {
      _logger.w("AppException caught: $error");
      final handledState = specificErrorHandler?.call(error);
      // Schedule state update with error handling.
      Future.microtask(() {
        state = handledState ??
            state.copyWith(
              status: AuthStatus.error,
              error: error.message,
            );
      });
    } catch (error) {
      _logger.e("Unexpected error in _executeAuthOperation: $error");
      // Update state for unexpected errors.
      Future.microtask(() {
        state = state.copyWith(
          status: AuthStatus.error,
          error: 'An unexpected error occurred',
        );
      });
    }
  }
}
