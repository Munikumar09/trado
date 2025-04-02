/*
Documentation:
---------------
Global Providers:
  This file defines application-wide providers.

  Providers:
    • secureStorageProvider:
        - Provides an instance of SecureStorageService.
        - Example: final storage = ref.read(secureStorageProvider);

    • navigatorKeyProvider:
        - Provides a GlobalKey<NavigatorState> for navigator access.
        - Example: final key = ref.read(navigatorKeyProvider);

    • dioProvider:
        - Initializes a Dio client using the provided secure storage.
        - Example: final dio = ref.read(dioProvider);
*/

// Code:
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/core/network/dio_client.dart';
import 'package:frontend/core/utils/exceptions/app_exceptions.dart';
import 'package:frontend/features/auth/application/services/token_storage_service.dart';

/// Provides an instance of SecureStorageService.
final secureStorageProvider = Provider((ref) => SecureStorageService());

/// Provides a GlobalKey for navigator access.
final navigatorKeyProvider = Provider<GlobalKey<NavigatorState>>(
  (ref) => GlobalKey<NavigatorState>(),
);

/// Provides a Dio instance.
/// Uses secure storage and handles initialization errors.
final dioProvider = Provider((ref) {
  try {
    final secureStorage = ref.read(secureStorageProvider);
    return DioClient(secureStorage, ref).dio;
  } catch (e, stackTrace) {
    // Throws a custom exception with error details.
    throw AppException(
        'Failed to initialize DIO client: ${e.toString()}', stackTrace);
  }
});
