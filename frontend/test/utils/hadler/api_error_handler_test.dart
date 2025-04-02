import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/core/utils/exceptions/app_exceptions.dart';
import 'package:frontend/core/utils/handlers/api_error_handler.dart';
import 'package:mocktail/mocktail.dart';

// Mock DioException
class MockDioException extends Mock implements DioException {}

// Mock Response
class MockResponse extends Mock implements Response {}

// Mock Request Options
class MockRequestOptions extends Mock implements RequestOptions {}

void main() {
  late MockDioException mockDioException;
  late MockResponse mockResponse;

  setUp(() {
    mockDioException = MockDioException();
    mockResponse = MockResponse();
    //Register fallback value.
    registerFallbackValue(RequestOptions(path: ''));
  });
  group('ApiErrorHandler', () {
    group('handleDioError', () {
      test('should return detail from response data if available', () {
        // Arrange
        const expectedMessage = 'Detailed error message';
        when(() => mockDioException.response).thenReturn(mockResponse);
        when(() => mockResponse.data).thenReturn({'detail': expectedMessage});
        when(() => mockDioException.type)
            .thenReturn(DioExceptionType.badResponse); // Set a type
        when(() => mockDioException.requestOptions)
            .thenReturn(RequestOptions(path: '')); //Always add this

        // Act
        final result = ApiErrorHandler.handleDioError(mockDioException);

        // Assert
        expect(result, expectedMessage);
      });

      test('should return default message if detail is not available', () {
        // Arrange
        when(() => mockDioException.response).thenReturn(mockResponse);
        when(() => mockResponse.data).thenReturn({}); // Empty data
        when(() => mockDioException.type)
            .thenReturn(DioExceptionType.connectionTimeout); // Set a type
        when(() => mockDioException.requestOptions)
            .thenReturn(RequestOptions(path: '')); //Always add this

        // Act
        final result = ApiErrorHandler.handleDioError(mockDioException);

        // Assert
        expect(result, 'Connection timeout');
      });
      test('should return default message if response is null', () {
        // Arrange
        when(() => mockDioException.response).thenReturn(null);
        when(() => mockDioException.message).thenReturn(null);
        when(() => mockDioException.type)
            .thenReturn(DioExceptionType.unknown); // Set a type
        when(() => mockDioException.requestOptions)
            .thenReturn(RequestOptions(path: '')); //Always add this

        // Act
        final result = ApiErrorHandler.handleDioError(mockDioException);

        // Assert
        expect(result, 'Unknown network error');
      });

      test('should return correct default message for each DioExceptionType',
          () {
        // Arrange
        final testCases = {
          DioExceptionType.connectionTimeout: 'Connection timeout',
          DioExceptionType.sendTimeout: 'Send timeout',
          DioExceptionType.receiveTimeout: 'Receive timeout',
          DioExceptionType.badCertificate: 'Bad Certificate',
          DioExceptionType.badResponse: 'Invalid server response',
          DioExceptionType.cancel: 'Request cancelled',
          DioExceptionType.connectionError: 'Connection error',
          DioExceptionType.unknown: 'Unknown network error',
        };
        when(() => mockDioException.requestOptions)
            .thenReturn(RequestOptions(path: '')); //Always add this
        when(() => mockDioException.response).thenReturn(null);
        when(() => mockDioException.message).thenReturn(null);

        testCases.forEach((dioExceptionType, expectedMessage) {
          // Arrange
          when(() => mockDioException.type).thenReturn(dioExceptionType);

          // Act
          final result = ApiErrorHandler.handleDioError(mockDioException);

          // Assert
          expect(result, expectedMessage,
              reason: 'Failed for type $dioExceptionType');
        });
      });

      test(
          'should return default message with status code for badResponse with response',
          () {
        // Arrange
        const statusCode = 400;
        when(() => mockDioException.response).thenReturn(mockResponse);
        when(() => mockResponse.data).thenReturn({}); // No detail
        when(() => mockResponse.statusCode).thenReturn(statusCode);
        when(() => mockDioException.type)
            .thenReturn(DioExceptionType.badResponse);
        when(() => mockDioException.requestOptions)
            .thenReturn(RequestOptions(path: '')); //Always add this

        // Act
        final result = ApiErrorHandler.handleDioError(mockDioException);

        // Assert
        expect(result, 'Invalid server response (Status code: $statusCode)');
      });
    });

    group('handleInvalidResponse', () {
      test('should return AppException with correct message for GET request',
          () {
        // Arrange
        final mockRequestOptions = MockRequestOptions();
        when(() => mockRequestOptions.method).thenReturn('GET');
        when(() => mockRequestOptions.path).thenReturn('/api/users');

        final mockResponse = MockResponse();
        when(() => mockResponse.statusCode).thenReturn(400);
        when(() => mockResponse.requestOptions).thenReturn(mockRequestOptions);

        // Act
        final appException =
            ApiErrorHandler.handleInvalidResponse(mockResponse);

        // Assert
        expect(appException, isA<AppException>());
        expect(
          appException.message,
          'Invalid response (400) for GET request to /api/users',
        );
      });

      test('should return AppException with correct message for POST request',
          () {
        // Arrange
        final mockRequestOptions = MockRequestOptions();
        when(() => mockRequestOptions.method).thenReturn('POST');
        when(() => mockRequestOptions.path).thenReturn('/api/login');

        final mockResponse = MockResponse();
        when(() => mockResponse.statusCode).thenReturn(500);
        when(() => mockResponse.requestOptions).thenReturn(mockRequestOptions);

        // Act
        final appException =
            ApiErrorHandler.handleInvalidResponse(mockResponse);

        // Assert
        expect(appException, isA<AppException>());
        expect(
          appException.message,
          'Invalid response (500) for POST request to /api/login',
        );
      });

      test(
          'should return AppException with correct message for different status code',
          () {
        // Arrange
        final mockRequestOptions = MockRequestOptions();
        when(() => mockRequestOptions.method).thenReturn('PUT');
        when(() => mockRequestOptions.path).thenReturn('/api/profile');

        final mockResponse = MockResponse();
        when(() => mockResponse.statusCode)
            .thenReturn(404); // Different status code
        when(() => mockResponse.requestOptions).thenReturn(mockRequestOptions);

        // Act
        final appException =
            ApiErrorHandler.handleInvalidResponse(mockResponse);

        // Assert
        expect(appException, isA<AppException>());
        expect(
          appException.message,
          'Invalid response (404) for PUT request to /api/profile',
        );
      });

      test('should return AppException with correct message for empty path',
          () {
        // Arrange
        final mockRequestOptions = MockRequestOptions();
        when(() => mockRequestOptions.method).thenReturn('DELETE');
        when(() => mockRequestOptions.path).thenReturn(''); // Empty path

        final mockResponse = MockResponse();
        when(() => mockResponse.statusCode).thenReturn(401);
        when(() => mockResponse.requestOptions).thenReturn(mockRequestOptions);

        // Act
        final appException =
            ApiErrorHandler.handleInvalidResponse(mockResponse);

        // Assert
        expect(appException, isA<AppException>());
        expect(
          appException.message,
          'Invalid response (401) for DELETE request to ', // Trailing space is fine
        );
      });
    });
  });
}
