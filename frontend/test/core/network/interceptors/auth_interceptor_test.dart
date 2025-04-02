import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/core/constants/storage_keys.dart';
import 'package:frontend/core/network/interceptor/auth_interceptor.dart';
import 'package:frontend/features/auth/application/providers/auth_providers.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';
import 'package:frontend/features/auth/application/state/auth_notifier.dart';
import 'package:frontend/features/auth/application/state/auth_state.dart';
import 'package:logger/logger.dart';
import 'package:mocktail/mocktail.dart';

// Mock classes
class MockSecureStorageService extends Mock implements SecureStorageService {}

class MockDio extends Mock implements Dio {}

class MockRequestOptions extends Mock implements RequestOptions {}

class MockResponse extends Mock implements Response {}

class MockRequestInterceptorHandler extends Mock
    implements RequestInterceptorHandler {} // Corrected

class MockErrorInterceptorHandler extends Mock
    implements ErrorInterceptorHandler {}

class MockRef extends Mock implements Ref {}

class MockLogger extends Mock implements Logger {}

class MockAuthNotifier extends Mock implements AuthNotifier {}

void main() {
  late AuthInterceptor authInterceptor;
  late MockSecureStorageService mockSecureStorage;
  late MockDio mockDio;
  late MockRequestOptions mockRequestOptions;
  late MockResponse mockResponse;
  late MockRequestInterceptorHandler mockRequestHandler; // Corrected type
  late MockErrorInterceptorHandler mockErrorHandler;
  late MockRef mockRef;
  late MockLogger mockLogger;
  late MockAuthNotifier mockAuthNotifier;

  setUpAll(() {
    registerFallbackValue(RequestOptions(path: ''));
    registerFallbackValue(MockLogger());
    registerFallbackValue(MockRef());
    registerFallbackValue(AuthState.initial());
    registerFallbackValue(Response(requestOptions: RequestOptions(path: '')));
    registerFallbackValue(MockRequestInterceptorHandler()); // Add this
  });

  setUp(() {
    mockSecureStorage = MockSecureStorageService();
    mockDio = MockDio();
    mockRequestOptions = MockRequestOptions();
    mockResponse = MockResponse();
    mockRequestHandler = MockRequestInterceptorHandler(); // Corrected
    mockErrorHandler = MockErrorInterceptorHandler();
    mockRef = MockRef();
    mockLogger = MockLogger(); // Initialize the mockLogger
    mockAuthNotifier = MockAuthNotifier(); // Create the mock instance

    // Correctly mock the *provider* to return our mock notifier.
    when(() => mockRef.read(authNotifierProvider.notifier))
        .thenReturn(mockAuthNotifier);

    // Now mock the logout() method on the *mock notifier*.
    when(() => mockAuthNotifier.logout()).thenAnswer((_) async {});

    authInterceptor = AuthInterceptor(mockSecureStorage, mockDio, mockRef);

    // Common setup for mockRequestOptions (avoids repetition)
    when(() => mockRequestOptions.headers).thenReturn({});
    when(() => mockRequestOptions.uri).thenReturn(Uri.parse(""));

    // Mock logger calls (good practice for completeness)
    when(() => mockLogger.d(any())).thenReturn(null);
    when(() => mockLogger.i(any())).thenReturn(null);
    when(() => mockLogger.w(any())).thenReturn(null);
    when(() => mockLogger.e(any(), [any, any])).thenReturn(null);
  });
  tearDown(() {
    reset(mockSecureStorage);
    reset(mockDio);
    reset(mockLogger);
    reset(mockRef);
    reset(mockRequestHandler);
    reset(mockErrorHandler);
    reset(mockRequestOptions);
    reset(mockResponse);
    reset(mockAuthNotifier);
  });

  group('AuthInterceptor', () {
    group('onRequest', () {
      test('adds authorization header if access token exists', () async {
        const accessToken = 'test_access_token';
        when(() => mockSecureStorage.getTokens()).thenAnswer((_) async =>
            (accessToken: accessToken, refreshToken: 'test_refresh_token'));
        when(() => mockRequestOptions.path).thenReturn('/some-endpoint');

        await authInterceptor.onRequest(mockRequestOptions, mockRequestHandler);

        expect(
            mockRequestOptions.headers['Authorization'], 'Bearer $accessToken');
        verify(() => mockRequestHandler.next(mockRequestOptions)).called(1);
        verify(() => mockSecureStorage.getTokens()).called(1);
      });

      test('does not add authorization header if access token is null',
          () async {
        when(() => mockSecureStorage.getTokens()).thenAnswer((_) async =>
            const (accessToken: null, refreshToken: 'test_refresh_token'));
        when(() => mockRequestOptions.path).thenReturn('/some-endpoint');

        await authInterceptor.onRequest(mockRequestOptions, mockRequestHandler);

        expect(mockRequestOptions.headers['Authorization'], isNull);
        verify(() => mockRequestHandler.next(mockRequestOptions)).called(1);
      });
    });

    group('onError', () {
      test('does nothing for non-401 errors', () async {
        final dioException = DioException(
          requestOptions: mockRequestOptions,
          response: mockResponse,
          type: DioExceptionType.badResponse,
          error: 'Some other error',
        );
        when(() => mockResponse.statusCode).thenReturn(400);

        await authInterceptor.onError(dioException, mockErrorHandler);

        verify(() => mockErrorHandler.next(dioException)).called(1);
        verifyNever(() => mockDio.post(any(),
            queryParameters:
                any(named: 'queryParameters'))); // Verify no refresh attempt
      });

      test('handles refresh token request 401, clears tokens, and logs out',
          () async {
        final dioException = DioException(
          requestOptions: mockRequestOptions,
          response: mockResponse,
          type: DioExceptionType.badResponse,
        );

        when(() => mockRequestOptions.path).thenReturn(
            AuthInterceptor.refreshTokenEndpoint); // Refresh token path
        when(() => mockResponse.statusCode).thenReturn(401);

        await authInterceptor.onError(dioException, mockErrorHandler);
        await pumpEventQueue();

        verify(() => mockRef.read(authNotifierProvider.notifier).logout())
            .called(1);
        verify(() => mockErrorHandler.reject(dioException)).called(1);
      });

      test('queues request if refresh is in progress', () async {
        final dioException = DioException(
          requestOptions: mockRequestOptions,
          response: mockResponse,
          type: DioExceptionType.badResponse,
        );
        when(() => mockResponse.statusCode).thenReturn(401);
        when(() => mockRequestOptions.path)
            .thenReturn('/some-other-endpoint'); // Different path

        // 1.  Set up a *successful* refresh token mock (to get past _attemptTokenRefresh)
        when(() => mockSecureStorage.getTokens()).thenAnswer((_) async =>
            const (accessToken: 'old_token', refreshToken: 'refresh_token'));
        when(() => mockDio.post(
              AuthInterceptor.refreshTokenEndpoint,
              queryParameters: any(named: 'queryParameters'),
              options: any(named: 'options'),
            )).thenAnswer((_) async => Response(
              requestOptions:
                  RequestOptions(path: AuthInterceptor.refreshTokenEndpoint),
              data: {'access_token': 'new_access_token'},
              statusCode: 200,
            ));
        when(() => mockSecureStorage.saveTokens(
            accessToken: any(named: 'accessToken'),
            refreshToken: any(named: 'refreshToken'))).thenAnswer((_) async {});

        // 2.  Call onError *once* to initiate the refresh.
        authInterceptor.onError(dioException, mockErrorHandler);

        // 3.  Call onError *again*, simulating a queued request.
        authInterceptor.onError(dioException, mockErrorHandler);
        await pumpEventQueue();

        // 4.  Verify that the refresh token request was only called once.
        verify(() => mockDio.post(AuthInterceptor.refreshTokenEndpoint,
                queryParameters: any(named: 'queryParameters')))
            .called(1); // ONLY ONCE

        verify(() => mockErrorHandler.reject(dioException))
            .called(2); // Called on the SECOND onError

        // // 5. Ensure other methods are not called when queued:
        verify(() => mockDio.fetch(any())).called(1);
        verifyNever(() => mockSecureStorage.clearTokens());
        verifyNever(() => mockRef.read(authNotifierProvider.notifier).logout());
      });

      test('handles successful token refresh and retries', () async {
        // Arrange
        final requestOptions = RequestOptions(path: '/test', headers: {});
        final dioException = DioException(
          requestOptions: requestOptions,
          response: Response(requestOptions: requestOptions, statusCode: 401),
        );
        const refreshToken = 'test_refresh_token';
        const newAccessToken = 'new_access_token';

        when(() => mockSecureStorage.getTokens()).thenAnswer((_) async =>
            (accessToken: 'old_token', refreshToken: refreshToken));

        // Mock the refresh token request.
        when(() => mockDio.post(
              AuthInterceptor.refreshTokenEndpoint,
              queryParameters: any(named: 'queryParameters'),
            )).thenAnswer((_) async => Response(
              requestOptions:
                  RequestOptions(path: AuthInterceptor.refreshTokenEndpoint),
              data: {'access_token': newAccessToken},
              statusCode: 200,
            ));

        // Mock saving the new tokens.
        when(() => mockSecureStorage.saveTokens(
              accessToken: newAccessToken,
              refreshToken: refreshToken,
            )).thenAnswer((_) async => Future.value());

        // Mock the retry (fetch) call.  Important: Use any() to match *any* options.
        when(() => mockDio.fetch<dynamic>(any()))
            .thenAnswer((invocation) async {
          final capturedOptions =
              invocation.positionalArguments.first as RequestOptions;
          return Response(
            requestOptions: capturedOptions, // Return the captured options
            data: {'message': 'success'},
            statusCode: 200,
          );
        });

        //Mock the resolve method of the handler
        when(() => mockErrorHandler.resolve(any())).thenAnswer((_) async => {});

        // Act
        await authInterceptor.onError(dioException, mockErrorHandler);
        await pumpEventQueue();
        // Assert
        verify(() => mockSecureStorage.getTokens()).called(1);
        verify(() => mockDio.post(AuthInterceptor.refreshTokenEndpoint,
            queryParameters: any(named: 'queryParameters'))).called(1);
        verify(() => mockSecureStorage.saveTokens(
            accessToken: newAccessToken, refreshToken: refreshToken)).called(1);

        // Verify that fetch was called with the *correct* options.
        final capturedFetchOptions = verify(() => mockDio.fetch(captureAny()))
            .captured
            .single as RequestOptions;
        expect(capturedFetchOptions.headers['Authorization'],
            'Bearer $newAccessToken');
        expect(capturedFetchOptions.path, '/test'); // Original path

        // Verify that the handler's resolve method was called.
        final capturedResolve =
            verify(() => mockErrorHandler.resolve(captureAny())).captured;
        expect(capturedResolve.length, equals(1));
        final capturedResponse = capturedResolve.single as Response;
        expect(capturedResponse.data, {
          'message': 'success'
        }); // Check the data from the *retried* request.
        expect(capturedResponse.statusCode, 200);

        // Verify no other actions were taken.
        verifyNever(() => mockSecureStorage.clearTokens());
        verifyNever(() => mockRef.read(authNotifierProvider.notifier).logout());
      });

      test(
          'handles refresh failure, clears tokens, and logs out (refresh token itself fails)',
          () async {
        final requestOptions = RequestOptions(path: '/test', headers: {});
        final handler = MockErrorInterceptorHandler();
        final dioException = DioException(
          requestOptions: requestOptions,
          response: Response(
              requestOptions: requestOptions,
              statusCode: 401,
              data: {'detail': 'Token Expired'}),
        );
        const refreshToken = 'test_refresh_token';
        when(() => mockSecureStorage.getTokens()).thenAnswer((_) async =>
            (accessToken: 'old_token', refreshToken: refreshToken));

        // Mock the refresh token request to *throw* a DioException (simulating a 401 from the refresh endpoint).
        when(() => mockDio.post(
              AuthInterceptor.refreshTokenEndpoint,
              queryParameters: any(named: 'queryParameters'),
            )).thenThrow(dioException);

        // Act
        await authInterceptor.onError(dioException, handler);

        // *** Add a delay here to allow async operations to complete ***
        await pumpEventQueue();

        // Assert
        verify(() => mockSecureStorage.getTokens()).called(1);
        verify(() => mockDio.post(AuthInterceptor.refreshTokenEndpoint,
            queryParameters: any(named: 'queryParameters'))).called(1);

        verify(() => mockRef.read(authNotifierProvider.notifier).logout())
            .called(1); //Verify call logout

        // Verify that the handler's reject method was called with the *original* error.
        verify(() => handler.reject(dioException)).called(1);

        // Verify no other actions were taken (no retries).
        verifyNever(() => mockDio.fetch(any()));
      });

      test(
          'onError - 401 error, successful token refresh, multiple queued requests',
          () async {
        // Arrange
        final requestOptions1 = RequestOptions(path: '/test1', headers: {});
        final requestOptions2 = RequestOptions(path: '/test2', headers: {});
        final handler1 = MockErrorInterceptorHandler();
        final handler2 = MockErrorInterceptorHandler();
        final dioException1 = DioException(
          requestOptions: requestOptions1,
          response: Response(
            requestOptions: requestOptions1,
            statusCode: 401,
          ),
        );
        final dioException2 = DioException(
          requestOptions: requestOptions2,
          response: Response(
            requestOptions: requestOptions2,
            statusCode: 401,
          ),
        );
        const refreshToken = 'test_refresh_token';
        const newAccessToken = 'new_access_token';

        when(() => mockSecureStorage.getTokens()).thenAnswer((_) async =>
            (accessToken: 'old_token', refreshToken: refreshToken));

        // Mock Dio.post to return a successful response WITH an access token
        when(() => mockDio.post(
              AuthInterceptor.refreshTokenEndpoint,
              queryParameters: {StorageKeys.refreshToken: refreshToken},
            )).thenAnswer((_) async => Response(
              requestOptions: RequestOptions(
                  path: AuthInterceptor.refreshTokenEndpoint, headers: {}),
              data: {
                StorageKeys.accessToken: newAccessToken
              }, // Include access token
              statusCode: 200, // Successful response
            ));
        when(() => mockSecureStorage.saveTokens(
                accessToken: newAccessToken, refreshToken: refreshToken))
            .thenAnswer((_) async => Future.value());

        // General mock for fetch to handle *both* requests.  Use `any()`.
        when(() => mockDio.fetch<dynamic>(any()))
            .thenAnswer((invocation) async {
          final capturedOptions =
              invocation.positionalArguments.first as RequestOptions;
          if (capturedOptions.path == '/test1') {
            return Response(
              requestOptions: capturedOptions, // Use captured options
              data: {'message': 'success1'},
              statusCode: 200,
            );
          } else if (capturedOptions.path == '/test2') {
            return Response(
              requestOptions: capturedOptions, // Use captured options
              data: {'message': 'success2'},
              statusCode: 200,
            );
          }
          throw Exception(
              'Unexpected path: ${capturedOptions.path}'); // Important
        });
        // Call onError for *both* requests *before* any verifications.
        authInterceptor.onError(dioException1, handler1); // Don't await here
        authInterceptor.onError(dioException2, handler2); // Don't await here

        // // Add the delay *after* both calls to onError.
        await pumpEventQueue();

        verify(() => mockSecureStorage.getTokens()).called(1);
        verify(() => mockDio.post(AuthInterceptor.refreshTokenEndpoint,
                queryParameters: {StorageKeys.refreshToken: refreshToken}))
            .called(1);
        verify(() => mockSecureStorage.saveTokens(
            accessToken: newAccessToken, refreshToken: refreshToken)).called(1);
        // expect(
        //     mockDio.options.headers['Authorization'], 'Bearer $newAccessToken'); // Removed
        expect(dioException2.requestOptions.headers['Authorization'],
            'Bearer $newAccessToken');

        // // Verify that fetch was called for *both* paths (using `any()`).
        verify(() => mockDio.fetch(any()))
            .called(2); // Called twice (for both requests)

        // // Capture and verify the arguments passed to handler1.resolve and handler2.resolve
        final captured1 = verify(() => handler1.resolve(captureAny())).captured;
        expect(captured1.length, 1);

        final capturedResponse1 = captured1.single as Response;
        expect(capturedResponse1.data, {'message': 'success1'});
        expect(capturedResponse1.statusCode, 200);
        expect(capturedResponse1.requestOptions.headers['Authorization'],
            'Bearer $newAccessToken');
        expect(capturedResponse1.requestOptions.path, '/test1');

        final captured2 = verify(() => handler2.resolve(captureAny())).captured;
        expect(captured2.length, 1);
        final capturedResponse2 = captured2.single as Response;
        expect(capturedResponse2.data, {'message': 'success2'});
        expect(capturedResponse2.statusCode, 200);
        expect(capturedResponse2.requestOptions.headers['Authorization'],
            'Bearer $newAccessToken');
        expect(capturedResponse2.requestOptions.path, '/test2');
      });
    });
  });
}
