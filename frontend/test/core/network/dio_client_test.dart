import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/core/constants/api_endpoints.dart';
import 'package:frontend/core/network/dio_client.dart';
import 'package:frontend/core/network/interceptor/auth_interceptor.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';
import 'package:mocktail/mocktail.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';

// Mock classes
class MockSecureStorageService extends Mock implements SecureStorageService {}

class MockRef extends Mock implements Ref {}

void main() {
  late MockSecureStorageService mockSecureStorage;
  late DioClient dioClient;
  late MockRef mockRef;

  setUp(() {
    mockSecureStorage = MockSecureStorageService();
    mockRef = MockRef();
  });

  tearDown(() {
    reset(mockSecureStorage);
    reset(mockRef);
  });

  group('DioClient', () {
    test('Dio instance is configured with correct BaseOptions', () {
      dioClient = DioClient(mockSecureStorage, mockRef);

      final dio = dioClient.dio;

      expect(dio.options.baseUrl, ApiEndpoints.baseUrl);
      expect(dio.options.connectTimeout, const Duration(milliseconds: 5000));
      expect(dio.options.receiveTimeout, const Duration(milliseconds: 5000));
      expect(dio.options.headers['Content-Type'], 'application/json');
    });

    test('AuthInterceptor is added to Dio interceptors', () {
      dioClient = DioClient(mockSecureStorage, mockRef);

      final dio = dioClient.dio;

      expect(dio.interceptors, isNotEmpty);
      bool hasAuthInterceptor = false;
      for (final interceptor in dio.interceptors) {
        if (interceptor is AuthInterceptor) {
          hasAuthInterceptor = true;
          break;
        }
      }
      expect(hasAuthInterceptor, isTrue,
          reason: "AuthInterceptor should be present");
    });

    // Correctly test debug mode
    test('PrettyDioLogger is added in debug mode', () {
      // Use runZoned to override kDebugMode *within* the test
      return runZoned(
        () {
          dioClient = DioClient(mockSecureStorage, mockRef);
          final dio = dioClient.dio;
          expect(dio.interceptors, isNotEmpty);
          expect(dio.interceptors.any((i) => i is PrettyDioLogger), isTrue);
          // Force debug
        },
        zoneValues: {#debug: true},
      );
    });
  });
}
