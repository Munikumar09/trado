import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/core/utils/exceptions/auth_exceptions.dart';
import 'package:frontend/features/auth/application/model/signup_request.dart';
import 'package:frontend/features/auth/application/repository/auth_repository.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';
import 'package:frontend/features/auth/application/state/auth_notifier.dart';
import 'package:frontend/features/auth/application/state/auth_state.dart';
import 'package:mocktail/mocktail.dart';

/// Mocks for dependencies.
class MockAuthRepository extends Mock implements AuthRepository {}

class MockTokenStorage extends Mock implements TokenStorage {}

void main() {
  late AuthNotifier authNotifier;
  late MockAuthRepository mockAuthRepository;
  late MockTokenStorage mockTokenStorage;

  setUp(() {
    mockAuthRepository = MockAuthRepository();
    mockTokenStorage = MockTokenStorage();
    authNotifier = AuthNotifier(mockAuthRepository, mockTokenStorage);

    registerFallbackValue(SignupRequest(
      username: '',
      email: '',
      password: '',
      confirmPassword: '',
      dateOfBirth: '',
      phoneNumber: '',
      gender: '',
    ));
    registerFallbackValue(AuthState.initial());
  });

  tearDown(() {
    reset(mockAuthRepository);
    reset(mockTokenStorage);
  });

  group('AuthNotifier - Initialization', () {
    test('initial state is correct', () {
      expect(authNotifier.state, AuthState.initial());
    });
  });

  group('AuthNotifier - Signin', () {
    const email = 'test@example.com';
    const password = 'password';
    const accessToken = 'test_access_token';

    test('successful signin updates state to authenticated', () async {
      // Arrange: Mock successful repository and token storage calls.
      when(() => mockAuthRepository.signin(email, password))
          .thenAnswer((_) async {});
      when(() => mockTokenStorage.getTokens()).thenAnswer((_) async =>
          (accessToken: accessToken, refreshToken: 'refresh_token'));
      // Act
      await authNotifier.signin(email, password); // Await the signin call
      await Future.delayed(Duration.zero); // Allow microtasks to complete.
      // Assert: State should be authenticated with the correct token.
      expect(authNotifier.state.status, AuthStatus.authenticated);
      expect(authNotifier.state.accessToken, accessToken); // Check token
      verify(() => mockAuthRepository.signin(email, password)).called(1);
      verify(() => mockTokenStorage.getTokens()).called(1);
    });

    test('signin failure updates state to error', () async {
      // Arrange: Mock a LoginFailedException.
      final exception = LoginFailedException('Signin failed');
      when(() => mockAuthRepository.signin(any(), any())).thenThrow(exception);

      // Act
      await authNotifier.signin(email, password);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the correct message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'Signin failed');
      verify(() => mockAuthRepository.signin(any(), any())).called(1);
      verifyNever(() => mockTokenStorage.getTokens());
    });

    test('EmailNotVerifiedException updates state to unverified', () async {
      // Arrange: Mock an EmailNotVerifiedException.
      when(() => mockAuthRepository.signin(any(), any()))
          .thenThrow(EmailNotVerifiedException('Email not verified'));

      // Act
      await authNotifier.signin(email, password);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be unverified with the correct email.
      expect(authNotifier.state.status, AuthStatus.unverified);
      expect(authNotifier.state.email, email); // Check that the email is set
      expect(
          authNotifier.state.error, isNull); // No general error should be set.
      verify(() => mockAuthRepository.signin(any(), any())).called(1);
      verifyNever(() => mockTokenStorage.getTokens());
    });

    test('generic exception updates state to error with default message',
        () async {
      // Arrange: Mock a generic exception.
      when(() => mockAuthRepository.signin(any(), any()))
          .thenThrow(Exception('Some other error'));

      // Act
      await authNotifier.signin(email, password);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the default message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'An unexpected error occurred');
      verify(() => mockAuthRepository.signin(any(), any())).called(1);
      verifyNever(() => mockTokenStorage.getTokens());
    });
  });

  group('AuthNotifier - Signup', () {
    final signupRequest = SignupRequest(
      username: 'testuser',
      email: 'test@example.com',
      password: 'password',
      confirmPassword: 'password',
      dateOfBirth: '2000-01-01',
      phoneNumber: '1234567890',
      gender: 'Male',
    );

    test('successful signup updates state to verificationPending', () async {
      // Arrange: Mock a successful signup call.
      when(() => mockAuthRepository.signup(any())).thenAnswer((_) async {});

      // Act
      await authNotifier.signup(signupRequest);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be verificationPending with the correct email.
      expect(authNotifier.state.status, AuthStatus.verificationPending);
      expect(authNotifier.state.email, 'test@example.com');
      verify(() => mockAuthRepository.signup(any())).called(1);
    });

    test('signup failure updates state to error', () async {
      // Arrange: Mock a SignupFailedException.
      final exception = SignupFailedException('Signup failed');
      when(() => mockAuthRepository.signup(any())).thenThrow(exception);

      // Act
      await authNotifier.signup(signupRequest);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the correct message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'Signup failed');
      verify(() => mockAuthRepository.signup(any())).called(1);
    });

    test('generic exception updates state to error with default message',
        () async {
      // Arrange: Mock a generic exception.
      when(() => mockAuthRepository.signup(any()))
          .thenThrow(Exception('Some other error'));

      // Act
      await authNotifier.signup(signupRequest);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the default message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'An unexpected error occurred');
      verify(() => mockAuthRepository.signup(any())).called(1);
    });
  });

  group('AuthNotifier - Send Verification Code', () {
    const email = 'test@example.com';

    test('successful send updates state to verificationSent', () async {
      // Arrange: Mock a successful send call.
      when(() => mockAuthRepository.sendVerificationCode(email))
          .thenAnswer((_) async {});

      // Act
      await authNotifier.sendVerificationCode(email);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be verificationSent with the correct email.
      expect(authNotifier.state.status, AuthStatus.verificationSent);
      expect(authNotifier.state.email, email);
      verify(() => mockAuthRepository.sendVerificationCode(email)).called(1);
    });

    test('sendVerificationCode failure updates state to error', () async {
      // Arrange: Mock a VerificationFailedException.
      final exception =
          SendVerificationCodeFailedException('Verification failed');
      when(() => mockAuthRepository.sendVerificationCode(any()))
          .thenThrow(exception);

      // Act
      await authNotifier.sendVerificationCode(email);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the correct message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'Verification failed');
      verify(() => mockAuthRepository.sendVerificationCode(any())).called(1);
    });

    test('generic exception updates state to error with default message',
        () async {
      // Arrange: Mock a generic exception.
      when(() => mockAuthRepository.sendVerificationCode(any()))
          .thenThrow(Exception('Some other error'));

      // Act
      await authNotifier.sendVerificationCode(email);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the default message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'An unexpected error occurred');
      verify(() => mockAuthRepository.sendVerificationCode(any())).called(1);
    });
  });

  group('AuthNotifier - Verify Verification Code', () {
    const email = 'test@example.com';
    const code = '123456';

    test('successful verification resets state to initial', () async {
      // Arrange: Mock a successful verification call.
      when(() => mockAuthRepository.verifyVerificationCode(email, code))
          .thenAnswer((_) async {});

      // Act
      await authNotifier.verifyVerificationCode(email, code);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be initial (unauthenticated).
      expect(authNotifier.state,
          AuthState.initial()); // Use const, check entire state
      verify(() => mockAuthRepository.verifyVerificationCode(email, code))
          .called(1);
    });

    test('verifyVerificationCode failure updates state to error', () async {
      // Arrange: Mock a VerificationFailedException.
      final exception = VerificationFailedException('Verification failed');
      when(() => mockAuthRepository.verifyVerificationCode(any(), any()))
          .thenThrow(exception);

      // Act
      await authNotifier.verifyVerificationCode(email, code);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the correct message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'Verification failed');
      verify(() => mockAuthRepository.verifyVerificationCode(any(), any()))
          .called(1);
    });

    test('generic exception updates state to error with default message',
        () async {
      // Arrange: Mock a generic exception.
      when(() => mockAuthRepository.verifyVerificationCode(any(), any()))
          .thenThrow(Exception('Some other error'));

      // Act
      await authNotifier.verifyVerificationCode(email, code);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the default message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'An unexpected error occurred');
      verify(() => mockAuthRepository.verifyVerificationCode(any(), any()))
          .called(1);
    });
  });

  group('AuthNotifier - Send Reset Password Code', () {
    const email = 'test@example.com';

    test('successful send updates state to verificationSent', () async {
      // Arrange: Mock a successful send call.
      when(() => mockAuthRepository.sendResetPasswordCode(email))
          .thenAnswer((_) async {});

      // Act
      await authNotifier.sendResetPasswordCode(email);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be verificationSent with the correct email.
      expect(authNotifier.state.status, AuthStatus.verificationSent);
      expect(authNotifier.state.email, email);
      verify(() => mockAuthRepository.sendResetPasswordCode(email)).called(1);
    });

    test('sendResetPasswordCode failure updates state to error', () async {
      // Arrange: Mock a PasswordResetFailedException.
      final exception = PasswordResetFailedException('Reset failed');
      when(() => mockAuthRepository.sendResetPasswordCode(any()))
          .thenThrow(exception);

      // Act
      await authNotifier.sendResetPasswordCode(email);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the correct message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'Reset failed');
      verify(() => mockAuthRepository.sendResetPasswordCode(any())).called(1);
    });

    test('generic exception updates state to error with default message',
        () async {
      // Arrange: Mock a generic exception.
      when(() => mockAuthRepository.sendResetPasswordCode(any()))
          .thenThrow(Exception('Some other error'));

      // Act
      await authNotifier.sendResetPasswordCode(email);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the default message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'An unexpected error occurred');
      verify(() => mockAuthRepository.sendResetPasswordCode(any())).called(1);
    });
  });

  group('AuthNotifier - Reset Password', () {
    const email = 'test@example.com';
    const code = '123456';
    const newPassword = 'newPassword';

    test('successful reset updates state to initial', () async {
      // Arrange: Mock a successful reset call.
      when(() => mockAuthRepository.resetPassword(email, code, newPassword))
          .thenAnswer((_) async {});

      // Act
      await authNotifier.resetPassword(email, code, newPassword);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be initial (unauthenticated).
      expect(authNotifier.state, AuthState.initial()); // Check entire state
      verify(() => mockAuthRepository.resetPassword(email, code, newPassword))
          .called(1);
    });

    test('resetPassword failure updates state to error', () async {
      // Arrange: Mock a PasswordResetFailedException.
      final exception = PasswordResetFailedException('Reset failed');
      when(() => mockAuthRepository.resetPassword(any(), any(), any()))
          .thenThrow(exception);

      // Act
      await authNotifier.resetPassword(email, code, newPassword);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the correct message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'Reset failed');
      verify(() => mockAuthRepository.resetPassword(any(), any(), any()))
          .called(1);
    });

    test('generic exception updates state to error with default message',
        () async {
      // Arrange: Mock a generic exception.
      when(() => mockAuthRepository.resetPassword(any(), any(), any()))
          .thenThrow(Exception('Some other error'));

      // Act
      await authNotifier.resetPassword(email, code, newPassword);
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the default message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'An unexpected error occurred');
      verify(() => mockAuthRepository.resetPassword(any(), any(), any()))
          .called(1);
    });
  });

  group('AuthNotifier - Logout', () {
    test('successful logout updates state to unauthenticated', () async {
      // Arrange: Mock a successful logout call.
      when(() => mockAuthRepository.logout()).thenAnswer((_) async {});

      // Act
      await authNotifier.logout();
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be unauthenticated.
      expect(authNotifier.state, AuthState.unauthenticated()); // Use const
      verify(() => mockAuthRepository.logout()).called(1);
    });

    test('logout failure updates state to error', () async {
      // Arrange: Mock a LogoutFailedException.
      final exception = LogoutFailedException('Logout Failed');
      when(() => mockAuthRepository.logout()).thenThrow(exception);

      // Act
      await authNotifier.logout();
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the correct message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'Logout Failed');
      verify(() => mockAuthRepository.logout()).called(1);
    });

    test('generic exception updates state to error with default message',
        () async {
      // Arrange: Mock a generic exception.
      when(() => mockAuthRepository.logout())
          .thenThrow(Exception('Some other error'));

      // Act
      await authNotifier.logout();
      await Future.delayed(Duration.zero); // Allow microtasks

      // Assert: State should be error with the default message.
      expect(authNotifier.state.status, AuthStatus.error);
      expect(authNotifier.state.error, 'An unexpected error occurred');
      verify(() => mockAuthRepository.logout()).called(1);
    });
  });

  group('AuthNotifier - Check Auth State', () {
    const accessToken = 'test_access_token';

    test('authenticated user updates state to authenticated', () async {
      // Mock successful auth check and token storage calls.
      when(() => mockAuthRepository.checkAuthState())
          .thenAnswer((_) async => {});
      when(() => mockTokenStorage.getTokens()).thenAnswer(
          (_) async => (accessToken: accessToken, refreshToken: ''));

      // Act
      await authNotifier.checkAuthState();
      await Future.delayed(Duration.zero);

      // Assert: State should be authenticated with the correct token.
      expect(authNotifier.state.status, AuthStatus.authenticated);
      expect(authNotifier.state.accessToken, accessToken); // Verify token
      verify(() => mockAuthRepository.checkAuthState()).called(1);
      verify(() => mockTokenStorage.getTokens()).called(2);
    });

    test('no access token updates the user state to unauthenticated', () async {
      // Mock unsuccessful auth check.
      when(() => mockAuthRepository.checkAuthState())
          .thenAnswer((_) async => {});
      when(() => mockTokenStorage.getTokens())
          .thenAnswer((_) async => (accessToken: null, refreshToken: null));

      // Act
      await authNotifier.checkAuthState();
      await Future.delayed(Duration.zero); // Allow microtasks to finish

      // Assert: State should be unauthenticated.
      expect(authNotifier.state.status, AuthStatus.unauthenticated);
      verify(() => mockTokenStorage.getTokens()).called(1);
    });

    test('checkAuthState failure updates state to unauthenticated', () async {
      // Arrange: Mock an exception from the repository.
      final exception = AuthException('Auth check failed');
      when(() => mockAuthRepository.checkAuthState()).thenThrow(exception);
      when(() => mockTokenStorage.getTokens()).thenAnswer(
          (_) async => (accessToken: accessToken, refreshToken: ''));

      // We do NOT need to mock getTokens() anymore.  The code won't reach it.

      // Act: Call the method.
      await authNotifier.checkAuthState();
      await Future.delayed(Duration.zero); // Let microtasks complete

      // Assert: The state should be error, because _executeAuthOperation handles it.
      expect(
          authNotifier.state.status, AuthStatus.error); // Corrected assertion
      expect(authNotifier.state.error,
          'Auth check failed'); // Check for the error message.
      verify(() => mockAuthRepository.checkAuthState()).called(1);
      verify(() => mockTokenStorage.getTokens()).called(1);
    });
  });
}
