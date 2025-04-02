import 'package:flutter/services.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/core/constants/storage_keys.dart';
import 'package:frontend/core/utils/exceptions/auth_exceptions.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';
import 'package:mocktail/mocktail.dart';

// Mock FlutterSecureStorage
class MockFlutterSecureStorage extends Mock implements FlutterSecureStorage {}

void main() {
  late SecureStorageService secureStorageService;
  late MockFlutterSecureStorage mockSecureStorage;

  setUp(() {
    mockSecureStorage = MockFlutterSecureStorage();
    secureStorageService =
        SecureStorageService(secureStorage: mockSecureStorage);
    registerFallbackValue(
        const AndroidOptions(encryptedSharedPreferences: true));
  });

  group('SecureStorageService', () {
    group('saveTokens', () {
      const accessToken = 'test_access_token';
      const refreshToken = 'test_refresh_token';

      test('should save tokens successfully', () async {
        // Arrange
        when(() => mockSecureStorage.write(
            key: any(named: 'key'),
            value: any(named: 'value'),
            iOptions: any(named: 'iOptions'),
            aOptions: any(named: 'aOptions'),
            lOptions: any(named: 'lOptions'),
            wOptions: any(named: 'wOptions'),
            mOptions: any(named: 'mOptions'),
            webOptions: any(named: 'webOptions'))).thenAnswer((_) async {});

        // Act
        await secureStorageService.saveTokens(
            accessToken: accessToken, refreshToken: refreshToken);

        // Assert
        verify(() => mockSecureStorage.write(
              key: StorageKeys.accessToken,
              value: accessToken,
            )).called(1);
        verify(() => mockSecureStorage.write(
              key: StorageKeys.refreshToken,
              value: refreshToken,
            )).called(1);
      });

      test('should throw TokenStorageException on PlatformException', () async {
        // Arrange
        when(() => mockSecureStorage.write(
                key: any(named: 'key'),
                value: any(named: 'value'),
                iOptions: any(named: 'iOptions'),
                aOptions: any(named: 'aOptions'),
                lOptions: any(named: 'lOptions'),
                wOptions: any(named: 'wOptions'),
                mOptions: any(named: 'mOptions'),
                webOptions: any(named: 'webOptions')))
            .thenThrow(PlatformException(code: 'TEST_ERROR'));

        // Act & Assert
        expect(
          () => secureStorageService.saveTokens(
              accessToken: accessToken, refreshToken: refreshToken),
          throwsA(isA<TokenStorageException>()),
        );
      });
    });

    group('getTokens', () {
      const accessToken = 'test_access_token';
      const refreshToken = 'test_refresh_token';

      test('should get tokens successfully', () async {
        // Arrange
        when(() => mockSecureStorage.read(
                key: any(named: 'key'),
                iOptions: any(named: 'iOptions'),
                aOptions: any(named: 'aOptions'),
                lOptions: any(named: 'lOptions'),
                wOptions: any(named: 'wOptions'),
                mOptions: any(named: 'mOptions'),
                webOptions: any(named: 'webOptions')))
            .thenAnswer((invocation) async {
          if (invocation.namedArguments[const Symbol('key')] ==
              StorageKeys.accessToken) {
            return accessToken;
          } else if (invocation.namedArguments[const Symbol('key')] ==
              StorageKeys.refreshToken) {
            return refreshToken;
          }
          return null;
        });

        // Act
        final result = await secureStorageService.getTokens();

        // Assert
        expect(result.accessToken, accessToken);
        expect(result.refreshToken, refreshToken);
        verify(() => mockSecureStorage.read(key: StorageKeys.accessToken))
            .called(1);
        verify(() => mockSecureStorage.read(key: StorageKeys.refreshToken))
            .called(1);
      });

      test('should return null values if tokens are not found', () async {
        // Arrange
        when(() => mockSecureStorage.read(
                key: any(named: 'key'),
                iOptions: any(named: 'iOptions'),
                aOptions: any(named: 'aOptions'),
                lOptions: any(named: 'lOptions'),
                wOptions: any(named: 'wOptions'),
                mOptions: any(named: 'mOptions'),
                webOptions: any(named: 'webOptions')))
            .thenAnswer((_) async => null);

        // Act
        final result = await secureStorageService.getTokens();

        // Assert
        expect(result.accessToken, isNull);
        expect(result.refreshToken, isNull);
        verify(() => mockSecureStorage.read(key: StorageKeys.accessToken))
            .called(1);
        verify(() => mockSecureStorage.read(key: StorageKeys.refreshToken))
            .called(1);
      });

      test('should throw TokenStorageException on PlatformException', () async {
        // Arrange
        when(() => mockSecureStorage.read(
                key: any(named: 'key'),
                iOptions: any(named: 'iOptions'),
                aOptions: any(named: 'aOptions'),
                lOptions: any(named: 'lOptions'),
                wOptions: any(named: 'wOptions'),
                mOptions: any(named: 'mOptions'),
                webOptions: any(named: 'webOptions')))
            .thenThrow(PlatformException(code: 'TEST_ERROR'));

        // Act & Assert
        expect(
          () => secureStorageService.getTokens(),
          throwsA(isA<TokenStorageException>()),
        );
      });
    });

    group('clearTokens', () {
      test('should clear tokens successfully', () async {
        // Arrange
        when(() => mockSecureStorage.delete(
            key: any(named: 'key'),
            iOptions: any(named: 'iOptions'),
            aOptions: any(named: 'aOptions'),
            lOptions: any(named: 'lOptions'),
            wOptions: any(named: 'wOptions'),
            mOptions: any(named: 'mOptions'),
            webOptions: any(named: 'webOptions'))).thenAnswer((_) async {});

        // Act
        await secureStorageService.clearTokens();

        // Assert
        verify(() => mockSecureStorage.delete(key: StorageKeys.accessToken))
            .called(1);
        verify(() => mockSecureStorage.delete(key: StorageKeys.refreshToken))
            .called(1);
      });

      test('should throw TokenStorageException on PlatformException', () async {
        // Arrange
        when(() => mockSecureStorage.delete(
                key: any(named: 'key'),
                iOptions: any(named: 'iOptions'),
                aOptions: any(named: 'aOptions'),
                lOptions: any(named: 'lOptions'),
                wOptions: any(named: 'wOptions'),
                mOptions: any(named: 'mOptions'),
                webOptions: any(named: 'webOptions')))
            .thenThrow(PlatformException(code: 'TEST_ERROR'));

        // Act & Assert
        expect(
          () => secureStorageService.clearTokens(),
          throwsA(isA<TokenStorageException>()),
        );
      });
    });

    group('parseAuthTokens', () {
      test('should parse tokens successfully', () {
        // Arrange
        const accessToken = 'test_access_token';
        const refreshToken = 'test_refresh_token';
        final data = {
          StorageKeys.accessToken: accessToken,
          StorageKeys.refreshToken: refreshToken,
        };

        // Act
        final result = secureStorageService.parseAuthTokens(data);

        // Assert
        expect(result.accessToken, accessToken);
        expect(result.refreshToken, refreshToken);
      });

      test('should throw FormatException if access token is missing', () {
        // Arrange
        final data = {
          StorageKeys.refreshToken: 'test_refresh_token',
        };

        // Act & Assert
        expect(() => secureStorageService.parseAuthTokens(data),
            throwsA(isA<FormatException>()));
      });

      test('should throw FormatException if refresh token is missing', () {
        // Arrange
        final data = {
          StorageKeys.accessToken: 'test_access_token',
        };

        // Act & Assert
        expect(() => secureStorageService.parseAuthTokens(data),
            throwsA(isA<FormatException>()));
      });
      test('should throw FormatException if both tokens are missing', () {
        // Arrange
        final data = <String, dynamic>{}; // Empty map

        // Act & Assert
        expect(() => secureStorageService.parseAuthTokens(data),
            throwsA(isA<FormatException>()));
      });
    });
  });
}
