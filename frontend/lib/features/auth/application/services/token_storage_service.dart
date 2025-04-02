/*
Documentation:
---------------
Classes:

1. TokenStorage (abstract):
   Description:
     Declares methods for secure token storage and management.
   Methods:
     • saveTokens({required accessToken, required refreshToken}):
         - Saves provided tokens into secure storage.
         - Example: await tokenStorage.saveTokens(accessToken: 'abc', refreshToken: 'def');
     • getTokens():
         - Retrieves the stored tokens as a record of optional accessToken and refreshToken.
         - Example: final tokens = await tokenStorage.getTokens();
     • clearTokens():
         - Clears all stored tokens.
         - Example: await tokenStorage.clearTokens();
     • parseAuthTokens(data):
         - Extracts tokens from a response data map; throws a FormatException if tokens are missing.
         - Example: final tokens = tokenStorage.parseAuthTokens(response.data);

2. SecureStorageService (implements TokenStorage):
   Description:
     Implements TokenStorage using FlutterSecureStorage for secure token I/O.
   Methods:
     - saveTokens:
         Writes tokens concurrently using FlutterSecureStorage.write.
     - getTokens:
         Reads tokens concurrently and returns a tuple of tokens.
     - clearTokens:
         Deletes the stored tokens concurrently.
     - parseAuthTokens:
         Validates and extracts tokens from a provided data map.
*/

// Code:
import 'package:flutter/services.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:frontend/core/constants/storage_keys.dart';
import 'package:frontend/core/utils/exceptions/auth_exceptions.dart';

/// TokenStorage abstract class declares methods for secure token storage and management.
abstract class TokenStorage {
  /// Saves access and refresh tokens.
  Future<void> saveTokens(
      {required String accessToken, required String refreshToken});

  /// Retrieves stored tokens as a record of accessToken and refreshToken.
  Future<({String? accessToken, String? refreshToken})> getTokens();

  /// Clears all stored tokens.
  Future<void> clearTokens();

  /// Parses authentication tokens from a data map. Throws a FormatException if tokens are missing.
  ({String accessToken, String refreshToken}) parseAuthTokens(
      Map<String, dynamic> data);
}

/// SecureStorageService securely stores tokens using FlutterSecureStorage.
class SecureStorageService implements TokenStorage {
  final FlutterSecureStorage _secureStorage;

  /// Constructs SecureStorageService, using an optional custom FlutterSecureStorage.
  SecureStorageService({FlutterSecureStorage? secureStorage})
      : _secureStorage = secureStorage ?? const FlutterSecureStorage();

  /// Saves tokens into secure storage.
  @override
  Future<void> saveTokens(
      {required String accessToken, required String refreshToken}) async {
    try {
      // Writing tokens concurrently into secure storage.
      await Future.wait([
        _secureStorage.write(key: StorageKeys.accessToken, value: accessToken),
        _secureStorage.write(
            key: StorageKeys.refreshToken, value: refreshToken),
      ]);
    } on PlatformException catch (e) {
      // Throws an exception if token saving fails.
      throw TokenStorageException('Failed to save tokens: ${e.message}');
    }
  }

  /// Retrieves tokens from secure storage.
  @override
  Future<({String? accessToken, String? refreshToken})> getTokens() async {
    try {
      // Reads tokens concurrently from secure storage.
      final results = await Future.wait([
        _secureStorage.read(key: StorageKeys.accessToken),
        _secureStorage.read(key: StorageKeys.refreshToken),
      ]);
      return (accessToken: results[0], refreshToken: results[1]);
    } on PlatformException catch (e) {
      // Throws an exception if token retrieval fails.
      throw TokenStorageException('Failed to read tokens: ${e.message}');
    }
  }

  /// Clears tokens from secure storage.
  @override
  Future<void> clearTokens() async {
    try {
      // Deletes tokens concurrently from secure storage.
      await Future.wait([
        _secureStorage.delete(key: StorageKeys.accessToken),
        _secureStorage.delete(key: StorageKeys.refreshToken),
      ]);
    } on PlatformException catch (e) {
      // Throws an exception if token deletion fails.
      throw TokenStorageException('Failed to clear tokens: ${e.message}');
    }
  }

  /// Parses and validates tokens from the provided data map.
  @override
  ({String accessToken, String refreshToken}) parseAuthTokens(
      Map<String, dynamic> data) {
    // Extract tokens from the data using keys.
    final accessToken = data[StorageKeys.accessToken] as String?;
    final refreshToken = data[StorageKeys.refreshToken] as String?;

    // Validate that both tokens are present.
    if (accessToken == null || refreshToken == null) {
      throw const FormatException('Invalid token format in response');
    }

    return (accessToken: accessToken, refreshToken: refreshToken);
  }
}
