import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:frontend/features/auth/application/providers/global_providers.dart';

final protectedDataProvider = FutureProvider<Map<String, dynamic>>((ref) async {
  final dio = ref.read(dioProvider); // Get Dio instance with auth interceptor
  try {
    final response =
        await dio.get('/authentication/protected-endpoint'); // API Call
    return response.data;
  } catch (e) {
    throw Exception("Failed to fetch protected data");
  }
});

class HomePage extends ConsumerWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final protectedData = ref.watch(protectedDataProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Home Page')),
      body: Center(
        child: protectedData.when(
          data: (data) => Text('Protected Data: ${data['message']}'),
          loading: () => const CircularProgressIndicator(),
          error: (error, _) => Text('Error: ${error.toString()}'),
        ),
      ),
    );
  }
}
