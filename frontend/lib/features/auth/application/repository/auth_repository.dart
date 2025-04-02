/*
Documentation:
---------------
Class: AuthRepository
Description:
  Handles all authentication API interactions including signup, login, verification,
  password reset, logout, and auth state checking. It uses a Dio client for HTTP requests,
  a token storage service to manage tokens, and an ApiCallHandler for error handling.

Methods:
  • signup(SignupRequest request):
      - Registers a new user.
      - Example: await authRepository.signup(signupRequest);
      
  • sendVerificationCode(String email):
      - Sends a verification code to the given email.
      - Example: await authRepository.sendVerificationCode('user@example.com');
      
  • verifyVerificationCode(String email, String emailOtp):
      - Verifies the provided OTP.
      - Example: await authRepository.verifyVerificationCode('user@example.com', '123456');
      
  • signin(String email, String password):
      - Signs in the user and stores tokens.
      - Example: await authRepository.signin('user@example.com', 'password123');
      
  • sendResetPasswordCode(String email):
      - Sends a reset password code.
      - Example: await authRepository.sendResetPasswordCode('user@example.com');
      
  • resetPassword(String email, String emailOtp, String newPassword):
      - Resets the user’s password.
      - Example: await authRepository.resetPassword('user@example.com', '123456', 'newPass');
      
  • logout():
      - Logs out the user by clearing tokens.
      - Example: await authRepository.logout();
      
  • checkAuthState():
      - Checks the current authentication state.
      - Example: await authRepository.checkAuthState();
*/

/* Code: */
import 'package:dio/dio.dart';
import 'package:frontend/core/constants/api_endpoints.dart';
import 'package:frontend/core/constants/app_strings.dart';
import 'package:frontend/core/utils/exceptions/auth_exceptions.dart';
import 'package:frontend/core/utils/handlers/api_call_handler.dart';
import 'package:frontend/features/auth/application/model/signup_request.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';

/// AuthRepository: Handles authentication API calls.
class AuthRepository {
  final Dio _dio;
  final SecureStorageService _tokenStorage;
  final ApiCallHandler _apiCallHandler;

  /// Constructs an AuthRepository using a Dio client, token storage service, and an API call handler.
  AuthRepository({
    required Dio dio,
    required SecureStorageService tokenStorage,
    required ApiCallHandler apiCallHandler,
  })  : _dio = dio,
        _tokenStorage = tokenStorage,
        _apiCallHandler = apiCallHandler;

  /// Registers a new user.
  Future<void> signup(SignupRequest request) async {
    await _apiCallHandler.handleApiCall<void>(
      call: () async {
        final response = await _dio.post(
          ApiEndpoints.signup,
          data: request.toJson(),
        );
        _apiCallHandler.validateResponse(response, successStatus: 201);
        return;
      },
      exception: (message) => SignupFailedException(message),
      operationName: 'Signup',
    );
  }

  /// Sends a verification code to the provided email.
  Future<void> sendVerificationCode(String email) async {
    await _apiCallHandler.handleApiCall<void>(
      call: () async {
        final response = await _dio.post(
          ApiEndpoints.sendVerification,
          queryParameters: {"email": email},
        );
        _apiCallHandler.validateResponse(response);
        return;
      },
      exception: (message) => SendVerificationCodeFailedException(message),
      operationName: 'Sending Verification Code',
    );
  }

  /// Verifies the email using the provided OTP.
  Future<void> verifyVerificationCode(String email, String emailOtp) async {
    await _apiCallHandler.handleApiCall<void>(
      call: () async {
        final response = await _dio.post(
          ApiEndpoints.verifyCode,
          data: {"verification_code": emailOtp, "email": email},
        );
        _apiCallHandler.validateResponse(response);
        return;
      },
      exception: (message) => VerificationFailedException(message),
      operationName: 'Email Verification',
    );
  }

  /// Signs in the user and stores tokens.
  Future<void> signin(String email, String password) async {
    return _apiCallHandler.handleApiCall<void>(
      call: () async {
        final response = await _dio.post(
          ApiEndpoints.signin,
          data: {'email': email, 'password': password},
        );
        final validatedResponse = _apiCallHandler.validateResponse(response);
        final tokens = _tokenStorage.parseAuthTokens(validatedResponse.data);
        await _tokenStorage.saveTokens(
            accessToken: tokens.accessToken, refreshToken: tokens.refreshToken);
        return;
      },
      exception: (message) {
        final lowerCaseMessage = message.toLowerCase();
        if (lowerCaseMessage.contains(AppStrings.userNotVerified)) {
          return EmailNotVerifiedException(message);
        }
        return LoginFailedException(message);
      },
      operationName: 'Login',
    );
  }

  /// Sends a reset password code to the specified email.
  Future<void> sendResetPasswordCode(String email) async {
    await _apiCallHandler.handleApiCall<void>(
      call: () async {
        final response = await _dio.post(
          ApiEndpoints.sendResetPasswordCode,
          queryParameters: {"email": email},
        );
        _apiCallHandler.validateResponse(response);
        return;
      },
      exception: (message) => PasswordResetFailedException(message),
      operationName: 'Sending Reset Password Code',
    );
  }

  /// Resets the password using the provided verification code.
  Future<void> resetPassword(
      String email, String emailOtp, String newPassword) async {
    await _apiCallHandler.handleApiCall<void>(
      call: () async {
        final response = await _dio.post(
          ApiEndpoints.resetPassword,
          data: {
            "email": email,
            "password": newPassword,
            "verification_code": emailOtp
          },
        );
        _apiCallHandler.validateResponse(response);
        return;
      },
      exception: (message) => PasswordResetFailedException(message),
      operationName: 'Reset Password',
    );
  }

  /// Logs out the user by clearing stored tokens.
  Future<void> logout() async {
    await _apiCallHandler.handleApiCall<void>(
      call: () async {
        await _tokenStorage.clearTokens();
        return;
      },
      exception: (message) => LogoutFailedException(message),
      operationName: 'Logout',
    );
  }

  /// Checks the current authentication state.
  Future<void> checkAuthState() async {
    await _apiCallHandler.handleApiCall<void>(
      call: () async {
        final response = await _dio.get(ApiEndpoints.protected);
        _apiCallHandler.validateResponse(response);
      },
      exception: (message) => AuthException(message),
      operationName: 'Check Auth State',
    );
  }
}
