import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/core/constants/api_endpoints.dart';
import 'package:frontend/core/utils/exceptions/app_exceptions.dart'; // Import AppException
import 'package:frontend/core/utils/exceptions/auth_exceptions.dart';
import 'package:frontend/core/utils/handlers/api_call_handler.dart';
import 'package:frontend/features/auth/application/model/signup_request.dart';
import 'package:frontend/features/auth/application/repository/auth_repository.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';
import 'package:mocktail/mocktail.dart';

// Mocks
class MockDio extends Mock implements Dio {}

class MockSecureStorageService extends Mock implements SecureStorageService {}

class MockApiCallHandler extends Mock implements ApiCallHandler {}

void main() {
  late AuthRepository authRepository;
  late MockDio mockDio;
  late MockSecureStorageService mockSecureStorageService;
  late MockApiCallHandler mockApiCallHandler;
  late AppException? capturedException;

  setUp(() {
    mockDio = MockDio();
    mockSecureStorageService = MockSecureStorageService();
    mockApiCallHandler = MockApiCallHandler();
    authRepository = AuthRepository(
      dio: mockDio,
      tokenStorage: mockSecureStorageService,
      apiCallHandler: mockApiCallHandler,
    );
    capturedException = null;
    when(() => mockApiCallHandler.handleApiCall<void>(
          call: any(named: 'call'),
          exception:
              any(named: 'exception'), // Capture the exception *function*
          operationName: any(named: 'operationName'),
        )).thenAnswer((invocation) async {
      try {
        await invocation.namedArguments[const Symbol('call')]();
      } on DioException catch (e) {
        // Get the exception function.
        final exceptionFunction =
            invocation.namedArguments[const Symbol('exception')] as AppException
                Function(String);
        // Get message from the exception *RESPONSE DATA*.
        final String errorMessage =
            e.response?.data['message']?.toString() ?? "Operation failed";

        // Call exception function with the actual error message *from the response*.
        capturedException = exceptionFunction(errorMessage);
        // Throw the created exception
        throw capturedException!; // We know it's not null.
      }
    });
    registerFallbackValue(Future<void>.value()); // For void returns
    registerFallbackValue(RequestOptions(path: ''));
    registerFallbackValue(Response(
        requestOptions: RequestOptions(path: ''),
        statusCode: 200)); // For Response
  });

  tearDown(() {
    // Reset mocks after each test.
    reset(mockDio);
    reset(mockSecureStorageService);
    reset(mockApiCallHandler);
    capturedException = null;
  });

  group('AuthRepository', () {
    group('signup', () {
      final signupRequest = SignupRequest(
        username: 'username',
        email: 'example@gmail.com',
        password: 'password',
        confirmPassword: 'password',
        dateOfBirth: '11/11/1999',
        phoneNumber: '1234567890',
        gender: 'male',
      );
      test('signup: should send correct request, handle success', () async {
        // Arrange
        final mockResponse = Response(
            requestOptions: RequestOptions(path: ApiEndpoints.signup),
            statusCode: 201);

        when(() => mockDio.post(
              ApiEndpoints.signup,
              data: any(named: 'data'),
            )).thenAnswer((invocation) async {
          final actualData = invocation.namedArguments[const Symbol('data')];
          expect(actualData, signupRequest.toJson());
          return mockResponse;
        });

        when(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 201)).thenReturn(mockResponse);

        // Act
        await authRepository.signup(signupRequest);

        // Assert
        verify(() => mockDio.post(
              ApiEndpoints.signup,
              data: signupRequest.toJson(),
            )).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
        verify(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 201)).called(1);
      });

      test('signup: should throw SignupFailedException on DioException',
          () async {
        // Arrange
        final dioException = DioException(
          requestOptions: RequestOptions(path: ApiEndpoints.signup),
          error: 'Network Error',
          type: DioExceptionType.connectionError,
        );

        when(() => mockDio.post(
              ApiEndpoints.signup,
              data: signupRequest.toJson(),
            )).thenThrow(dioException);

        // Act and Assert
        await expectLater(
          () => authRepository.signup(signupRequest),
          throwsA(isA<SignupFailedException>()),
        );

        // Verify that the *captured* exception is the correct type.
        expect(capturedException, isA<SignupFailedException>());
        expect(
            (capturedException as SignupFailedException).message,
            equals(
                dioException.message ?? "Operation failed")); // Check message

        verify(() => mockDio.post(
              ApiEndpoints.signup,
              data: signupRequest.toJson(),
            )).called(1);

        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);

        verifyNever(() => mockApiCallHandler.validateResponse(any(),
            successStatus: any(named: 'successStatus')));
      });
    });
    group('signin', () {
      final email = 'test@example.com';
      final password = 'password';
      final mockAccessToken = 'mockAccessToken';
      final mockRefreshToken = 'mockRefreshToken';

      test('signin: should make a successful signin and save tokens', () async {
        // Arrange:  Set up the conditions for the test.

        // Mock the API response for a successful signin.
        final mockResponse = Response(
            requestOptions: RequestOptions(path: ApiEndpoints.signin),
            statusCode: 200,
            data: {
              'access_token': mockAccessToken,
              'refresh_token': mockRefreshToken
            });

        // Mock Dio.post to return the successful response.
        when(() => mockDio.post(ApiEndpoints.signin,
                data: {'email': email, 'password': password}))
            .thenAnswer((_) async => mockResponse);

        // Mock validateResponse to return the mock response.
        when(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).thenReturn(mockResponse);

        // Mock the secure storage service to save the tokens.
        when(() => mockSecureStorageService.saveTokens(
            accessToken: mockAccessToken,
            refreshToken: mockRefreshToken)).thenAnswer((_) async {});

        // Mock the secure storage service to parse auth tokens.
        when(() => mockSecureStorageService.parseAuthTokens({
                  'access_token': mockAccessToken,
                  'refresh_token': mockRefreshToken
                }))
            .thenReturn(
                (accessToken: mockAccessToken, refreshToken: mockRefreshToken));

        // Act:  Execute the code under test.
        await authRepository.signin(email, password);

        // Assert:  Verify that the expected behavior occurred.
        verify(() => mockDio.post(ApiEndpoints.signin,
            data: {'email': email, 'password': password})).called(1);
        verify(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).called(1);
        verify(() => mockSecureStorageService.saveTokens(
            accessToken: mockAccessToken,
            refreshToken: mockRefreshToken)).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
      });

      test('signin: should throw LoginFailedException when signin fails',
          () async {
        // Arrange
        final dioException = DioException(
          requestOptions: RequestOptions(path: ApiEndpoints.signin),
          error: 'Invalid credentials',
          type: DioExceptionType.badResponse,
        );

        when(() => mockDio.post(
              ApiEndpoints.signin,
              data: {'email': email, 'password': password},
            )).thenThrow(dioException);

        // Act and Assert
        await expectLater(
          () => authRepository.signin(email, password),
          throwsA(isA<LoginFailedException>()),
        );

        expect(capturedException, isA<LoginFailedException>());
        expect(
            (capturedException as LoginFailedException).message,
            equals(
                dioException.message ?? "Operation failed")); // Check message
        verify(() => mockDio.post(
              ApiEndpoints.signin,
              data: {'email': email, 'password': password},
            )).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
        verifyNever(() => mockSecureStorageService.saveTokens(
            accessToken: any(named: "accessToken"),
            refreshToken: any(named: "refreshToken")));
      });

      test(
          'signin: should throw EmailNotVerifiedException when the API returns a 403 with specific message',
          () async {
        // Arrange
        final email = 'test@example.com';
        final password = 'password';
        final expectedMessage =
            'Email not verified'; // Define expected message.
        final dioException = DioException(
          requestOptions: RequestOptions(path: ApiEndpoints.signin),
          response: Response(
            requestOptions: RequestOptions(path: ApiEndpoints.signin),
            statusCode: 403,
            data: {'message': expectedMessage}, // Use the expected message
          ),
          type: DioExceptionType.badResponse,
        );

        when(() => mockDio.post(
              ApiEndpoints.signin,
              data: {'email': email, 'password': password},
            )).thenThrow(dioException);

        // Act and Assert
        await expectLater(
          () => authRepository.signin(email, password),
          throwsA(isA<EmailNotVerifiedException>()),
        );

        // Verify the *captured* exception, and check the MESSAGE.
        expect(capturedException, isA<EmailNotVerifiedException>());
        expect((capturedException as EmailNotVerifiedException).message,
            equals(expectedMessage)); // Exact match

        verify(() => mockDio.post(
              ApiEndpoints.signin,
              data: {'email': email, 'password': password},
            )).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
        verifyNever(() => mockSecureStorageService.saveTokens(
            accessToken: any(named: "accessToken"),
            refreshToken: any(named: "refreshToken")));
      });
    });
    group('sendVerificationCode', () {
      final email = 'test@example.com';

      test('sendVerificationCode: should make a successful call', () async {
        // Arrange:  Set up the conditions for the test.
        final email = 'test@example.com'; // Define email here
        final mockResponse = Response(
            requestOptions: RequestOptions(path: ApiEndpoints.sendVerification),
            statusCode: 200);

        when(() => mockDio.post(ApiEndpoints.sendVerification,
                queryParameters: {'email': email}))
            .thenAnswer((_) async => mockResponse);

        when(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).thenReturn(mockResponse);

        // Act: Execute the code being tested.
        await authRepository.sendVerificationCode(email);

        // Assert: Check that the methods were called.
        verify(() => mockDio.post(ApiEndpoints.sendVerification,
                queryParameters: {'email': email})) // Changed to data:
            .called(1);
        verify(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              // Verify with void
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
      });

      test(
          'sendVerificationCode: should throw VerificationFailedException on DioException',
          () async {
        // Arrange
        final dioException = DioException(
          requestOptions: RequestOptions(path: ApiEndpoints.sendVerification),
          error: 'Failed to send code',
          type: DioExceptionType.badResponse,
        );

        when(() => mockDio.post(
              ApiEndpoints.sendVerification,
              queryParameters: {'email': email},
            )).thenThrow(dioException);

        // Act and Assert
        await expectLater(
          () => authRepository.sendVerificationCode(email),
          throwsA(isA<SendVerificationCodeFailedException>()),
        );
        expect(capturedException, isA<SendVerificationCodeFailedException>());
        expect(
            (capturedException as SendVerificationCodeFailedException).message,
            equals(
                dioException.message ?? "Operation failed")); // Check message
        verify(() => mockDio.post(
              ApiEndpoints.sendVerification,
              queryParameters: {'email': email},
            )).called(1);

        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);

        verifyNever(() => mockApiCallHandler.validateResponse(any(),
            successStatus: any(named: 'successStatus')));
      });
    });

    group('verifyVerificationCode', () {
      final email = 'test@example.com';
      final code = '123456';

      test('verifyVerificationCode: should make a successful call', () async {
        // Arrange: Setup mocks for the test.
        final mockResponse = Response(
            requestOptions: RequestOptions(path: ApiEndpoints.verifyCode),
            statusCode: 200);

        when(() => mockDio.post(ApiEndpoints.verifyCode,
                data: {'verification_code': code, 'email': email}))
            .thenAnswer((_) async => mockResponse);

        when(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).thenReturn(mockResponse);

        // Act: Execute the code being tested.
        await authRepository.verifyVerificationCode(email, code);

        // Assert: Check that the methods were called.
        verify(() => mockDio.post(ApiEndpoints.verifyCode,
            data: {'verification_code': code, 'email': email})).called(1);
        verify(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
      });

      test(
          'verifyVerificationCode: should throw VerificationFailedException when the request fails',
          () async {
        // Arrange
        final dioException = DioException(
          requestOptions: RequestOptions(path: ApiEndpoints.verifyCode),
          error: 'Invalid code',
          type: DioExceptionType.badResponse,
        );

        when(() => mockDio.post(
              ApiEndpoints.verifyCode,
              data: {'verification_code': code, 'email': email},
            )).thenThrow(dioException);

        // Act and Assert
        await expectLater(
          () => authRepository.verifyVerificationCode(email, code),
          throwsA(isA<VerificationFailedException>()),
        );

        expect(capturedException, isA<VerificationFailedException>());
        expect(
            (capturedException as VerificationFailedException).message,
            equals(
                dioException.message ?? "Operation failed")); // Check message

        verify(() => mockDio.post(
              ApiEndpoints.verifyCode,
              data: {'email': email, 'verification_code': code},
            )).called(1);

        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);

        verifyNever(() => mockApiCallHandler.validateResponse(any(),
            successStatus: any(named: 'successStatus')));
      });
    });

    group('sendResetPasswordCode', () {
      final email = 'test@example.com';

      test('sendResetPasswordCode: should make a successful call', () async {
        // Arrange:  Set up mocks for the test.
        final mockResponse = Response(
            requestOptions:
                RequestOptions(path: ApiEndpoints.sendResetPasswordCode),
            statusCode: 200);

        when(() => mockDio.post(ApiEndpoints.sendResetPasswordCode,
                queryParameters: {'email': email}))
            .thenAnswer((_) async => mockResponse);

        when(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).thenReturn(mockResponse);

        // Act: Execute the code being tested.
        await authRepository.sendResetPasswordCode(email);

        // Assert: Check that the methods were called.
        verify(() => mockDio.post(ApiEndpoints.sendResetPasswordCode,
            queryParameters: {'email': email})).called(1);
        verify(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
      });

      test(
          'sendResetPasswordCode: should throw PasswordResetFailedException when the request fails',
          () async {
        // Arrange
        final dioException = DioException(
          requestOptions:
              RequestOptions(path: ApiEndpoints.sendResetPasswordCode),
          error: 'Failed to send reset code',
          type: DioExceptionType.badResponse,
        );

        when(() => mockDio.post(
              ApiEndpoints.sendResetPasswordCode,
              queryParameters: {'email': email},
            )).thenThrow(dioException);

        // Act and Assert
        await expectLater(
          () => authRepository.sendResetPasswordCode(email),
          throwsA(isA<PasswordResetFailedException>()),
        );

        expect(capturedException, isA<PasswordResetFailedException>());
        expect(
            (capturedException as PasswordResetFailedException).message,
            equals(
                dioException.message ?? "Operation failed")); // Check message
        verify(() => mockDio.post(
              ApiEndpoints.sendResetPasswordCode,
              queryParameters: {'email': email},
            )).called(1);

        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);

        verifyNever(() => mockApiCallHandler.validateResponse(any(),
            successStatus: any(named: 'successStatus')));
      });
    });

    group('resetPassword', () {
      final email = 'test@example.com';
      final code = '123456';
      final newPassword = 'newPassword';

      test('resetPassword: should make a successful call', () async {
        // Arrange: Setup mocks for the test
        final mockResponse = Response(
            requestOptions: RequestOptions(path: ApiEndpoints.resetPassword),
            statusCode: 200);

        when(() => mockDio.post(ApiEndpoints.resetPassword, data: {
              'email': email,
              'verification_code': code,
              'password': newPassword
            })).thenAnswer((_) async => mockResponse);

        when(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).thenReturn(mockResponse);

        // Act: Execute the code being tested.
        await authRepository.resetPassword(email, code, newPassword);

        // Assert: Check that the methods were called.
        verify(() => mockDio.post(ApiEndpoints.resetPassword, data: {
              'email': email,
              'verification_code': code,
              'password': newPassword
            })).called(1);
        verify(() => mockApiCallHandler.validateResponse(mockResponse,
            successStatus: 200)).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
      });

      test(
          'resetPassword: should throw PasswordResetFailedException when the request fails',
          () async {
        // Arrange
        final dioException = DioException(
          requestOptions: RequestOptions(path: ApiEndpoints.resetPassword),
          error: 'Failed to reset password',
          type: DioExceptionType.badResponse,
        );

        when(() => mockDio.post(
              ApiEndpoints.resetPassword,
              data: {
                'email': email,
                'verification_code': code,
                'password': newPassword
              },
            )).thenThrow(dioException);

        // Act and Assert
        await expectLater(
          () => authRepository.resetPassword(email, code, newPassword),
          throwsA(isA<PasswordResetFailedException>()),
        );

        expect(capturedException, isA<PasswordResetFailedException>());
        expect(
            (capturedException as PasswordResetFailedException).message,
            equals(
                dioException.message ?? "Operation failed")); // Check message

        verify(() => mockDio.post(
              ApiEndpoints.resetPassword,
              data: {
                'email': email,
                'verification_code': code,
                'password': newPassword
              },
            )).called(1);

        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);

        verifyNever(() => mockApiCallHandler.validateResponse(any(),
            successStatus: any(named: 'successStatus')));
      });
    });

    group('logout', () {
      test('logout: should make a successful call and clear tokens', () async {
        // Arrange: Setup mocks for the test

        when(() => mockSecureStorageService.clearTokens())
            .thenAnswer((_) async {});

        // Act: Execute the code being tested.
        await authRepository.logout();

        // Assert: Check that the methods were called.

        verify(() => mockSecureStorageService.clearTokens()).called(1);
        verify(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).called(1);
      });

      test('logout: should throw LogoutFailedException when clearTokens fails',
          () async {
        // Arrange: Setup mocks for the test.
        final exceptionMessage = 'Failed to clear tokens';
        final mockException = TokenStorageException(exceptionMessage);

        // Mock secureStorageService.clearTokens() to throw the exception.
        when(() => mockSecureStorageService.clearTokens())
            .thenThrow(mockException);

        // Mock handleApiCall to correctly handle the exception.
        when(() => mockApiCallHandler.handleApiCall<void>(
              call: any(named: 'call'),
              exception: any(named: 'exception'),
              operationName: any(named: 'operationName'),
            )).thenAnswer((invocation) async {
          try {
            await invocation.namedArguments[const Symbol('call')]();
          } on TokenStorageException catch (e) {
            // Catch the *correct* exception
            final exceptionFunction =
                invocation.namedArguments[const Symbol('exception')]
                    as AppException Function(String);
            throw exceptionFunction(e.message); // Use e.message, not a default.
          }

          // Act and Assert
          await expectLater(
            () => authRepository.logout(),
            throwsA(isA<LogoutFailedException>()),
          );

          verify(() => mockApiCallHandler.handleApiCall<void>(
                call: any(named: 'call'),
                exception: any(named: 'exception'),
                operationName: any(named: 'operationName'),
              )).called(1);
          verify(() => mockSecureStorageService.clearTokens()).called(1);
        });
      });
      group('checkAuthState', () {
        test('checkAuthState: should make a successful call', () async {
          // Arrange: Mock getTokens to return valid tokens.  We *still* need this.

          final mockResponse = Response(
              requestOptions: RequestOptions(path: ApiEndpoints.protected),
              statusCode: 200);
          // Mock the Dio.get call (it's a GET, not a POST).
          when(() => mockDio.get(ApiEndpoints.protected)) // Use queryParameters
              .thenAnswer((_) async => mockResponse);

          // Mock validateResponse (good practice, even if it's simple).
          when(() => mockApiCallHandler.validateResponse(any(),
              successStatus: 200)).thenReturn(mockResponse);

          // Act: Execute the code being tested
          await authRepository.checkAuthState();

          verify(() => mockDio.get(ApiEndpoints.protected))
              .called(1); // Corrected verification
          verify(() => mockApiCallHandler.handleApiCall<void>(
                call: any(named: 'call'),
                exception: any(named: 'exception'),
                operationName: any(named: 'operationName'),
              )).called(1);
          verify(() => mockApiCallHandler.validateResponse(any(),
              successStatus: 200)).called(1);
        });

        test('checkAuthState: should throw AuthException on DioException',
            () async {
          // Arrange
          final dioException = DioException(
            requestOptions: RequestOptions(path: ApiEndpoints.protected),
            error: 'checking auth state failed',
            type: DioExceptionType.badResponse,
            response: Response(
              data: {'message': 'Unauthorized'},
              // Include a response for more realistic error
              requestOptions: RequestOptions(path: ApiEndpoints.protected),
              statusCode: 401, // Unauthorized
            ),
          );

          // Mock Dio.get to *throw* the DioException.
          when(() => mockDio.get(ApiEndpoints.protected)) // Use queryParameters
              .thenThrow(dioException);

          // Act and Assert
          await expectLater(
            () => authRepository.checkAuthState(),
            throwsA(isA<AuthException>()),
          );

          expect(capturedException, isA<AuthException>());
          expect((capturedException as AuthException).message,
              equals(dioException.message ?? "Unauthorized")); // Check message

          verify(() => mockDio.get(ApiEndpoints.protected))
              .called(1); // Correct verification.
          verify(() => mockApiCallHandler.handleApiCall<void>(
                call: any(named: 'call'),
                exception: any(named: 'exception'),
                operationName: any(named: 'operationName'),
              )).called(1);
        });
      });
    });
  });
}
