/*
Documentation:
---------------
Auth Providers:
  This file bundles authentication-related providers.

  Providers:
    • apiCallHandlerProvider:
        - Provides an instance of ApiCallHandler for standardized API call execution.
        - Example: final apiHandler = ref.read(apiCallHandlerProvider);

    • authRepositoryProvider:
        - Provides an instance of AuthRepository to handle authentication API interactions.
        - Example: final authRepo = ref.read(authRepositoryProvider);

    • authNotifierProvider:
        - A StateNotifierProvider which manages authentication state using AuthNotifier.
        - Example: ref.watch(authNotifierProvider);
*/

//Code:
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/core/utils/handlers/api_call_handler.dart';
import 'package:frontend/features/auth/application/providers/global_providers.dart';
import 'package:frontend/features/auth/application/repository/auth_repository.dart';
import 'package:frontend/features/auth/application/state/auth_notifier.dart';
import 'package:frontend/features/auth/application/state/auth_state.dart';

/// Provides an instance of ApiCallHandler.
final apiCallHandlerProvider = Provider((ref) => ApiCallHandler());

/// Provides an instance of AuthRepository using Dio, SecureStorage, and ApiCallHandler.
final authRepositoryProvider = Provider((ref) {
  final dio = ref.read(dioProvider);
  final secureStorage = ref.read(secureStorageProvider);
  final apiCallHandler = ref.read(apiCallHandlerProvider);
  return AuthRepository(
      dio: dio, tokenStorage: secureStorage, apiCallHandler: apiCallHandler);
});

/// Provides a StateNotifier for managing authentication state.
final authNotifierProvider =
    StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  final authRepository = ref.watch(authRepositoryProvider);
  final secureStorage = ref.watch(secureStorageProvider);
  return AuthNotifier(authRepository, secureStorage);
});
