import 'package:flutter/material.dart';
import 'package:frontend/features/stock_ticker/application/stock_ticker_provider.dart';
import 'package:frontend/features/stock_ticker/domain/stock_data.dart';
import 'package:intl/intl.dart'; // For date formatting
import 'package:provider/provider.dart';

class StockTickerScreen extends StatefulWidget {
  const StockTickerScreen({super.key});

  @override
  State<StockTickerScreen> createState() => _StockTickerScreenState();
}

class _StockTickerScreenState extends State<StockTickerScreen> {
  final TextEditingController _tokenController = TextEditingController();

  @override
  void initState() {
    super.initState();
    // Attempt to connect automatically when the screen is initialized
    // Use addPostFrameCallback to ensure Provider is available
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // Check if already connected or connecting before attempting
      final provider = Provider.of<StockTickerProvider>(context, listen: false);
      if (provider.connectionStatus != ConnectionStatus.connected &&
          provider.connectionStatus != ConnectionStatus.connecting) {
         provider.connect();
      }
    });
  }

  @override
  void dispose() {
    _tokenController.dispose();
    // Consider if provider disconnect should happen here or be managed globally
    // Provider.of<StockTickerProvider>(context, listen: false).disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Stock Ticker'),
        actions: [
          // Connection Status Indicator
          Consumer<StockTickerProvider>(
            builder: (context, provider, child) {
              IconData icon;
              Color color;
              String tooltip;
              switch (provider.connectionStatus) {
                case ConnectionStatus.connected:
                  icon = Icons.cloud_done;
                  color = Colors.green;
                  tooltip = 'Connected';
                  break;
                case ConnectionStatus.connecting:
                  icon = Icons.cloud_sync;
                  color = Colors.orange;
                  tooltip = 'Connecting...';
                  break;
                case ConnectionStatus.disconnected:
                  icon = Icons.cloud_off;
                  color = Colors.grey;
                  tooltip = 'Disconnected';
                  break;
                case ConnectionStatus.error:
                  icon = Icons.cloud_off;
                  color = Colors.red;
                  tooltip = 'Error: ${provider.errorMessage ?? 'Unknown'}';
                  break;
              }
              return IconButton(
                icon: Icon(icon, color: color),
                tooltip: tooltip,
                onPressed: () {
                  // Allow manual connect/disconnect
                  if (provider.connectionStatus == ConnectionStatus.connected) {
                    provider.disconnect();
                  } else if (provider.connectionStatus == ConnectionStatus.disconnected || provider.connectionStatus == ConnectionStatus.error) {
                     provider.connect();
                  }
                },
              );
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Subscription Input Area
            _buildSubscriptionInput(context),
            const SizedBox(height: 10),
            // Error Message Display
            Consumer<StockTickerProvider>(
              builder: (context, provider, child) {
                if (provider.connectionStatus == ConnectionStatus.error && provider.errorMessage != null) {
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 8.0),
                    child: Text(
                      'Error: ${provider.errorMessage}',
                      style: TextStyle(color: Theme.of(context).colorScheme.error),
                      textAlign: TextAlign.center,
                    ),
                  );
                }
                return const SizedBox.shrink(); // No error, show nothing
              },
            ),
            const SizedBox(height: 10),
            const Text(
              'Subscribed Stocks:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const Divider(),
            // List of Subscribed Stocks
            Expanded(
              child: Consumer<StockTickerProvider>(
                builder: (context, provider, child) {
                  if (provider.subscribedTokens.isEmpty) {
                    return const Center(child: Text('No stocks subscribed.'));
                  }
                  // Display subscribed tokens and their latest data
                  return ListView.builder(
                    itemCount: provider.subscribedTokens.length,
                    itemBuilder: (context, index) {
                      final token = provider.subscribedTokens.elementAt(index);
                      final stockData = provider.stockDataMap[token];
                      return _buildStockListItem(context, token, stockData);
                    },
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSubscriptionInput(BuildContext context) {
    final provider = Provider.of<StockTickerProvider>(context, listen: false); // Use listen:false for actions
    return Row(
      children: [
        Expanded(
          child: TextField(
            controller: _tokenController,
            decoration: const InputDecoration(
              labelText: 'Stock Token (e.g., RELIANCE)',
              border: OutlineInputBorder(),
            ),
            onSubmitted: (value) { // Allow submitting with Enter key
               if (value.isNotEmpty) {
                 provider.subscribe(value);
                 _tokenController.clear();
               }
            },
          ),
        ),
        const SizedBox(width: 8),
        ElevatedButton(
          onPressed: () {
            final token = _tokenController.text.trim();
            if (token.isNotEmpty) {
              provider.subscribe(token);
              _tokenController.clear();
               FocusScope.of(context).unfocus(); // Dismiss keyboard
            }
          },
          child: const Text('Subscribe'),
        ),
      ],
    );
  }

  Widget _buildStockListItem(BuildContext context, String token, StockData? stockData) {
     final provider = Provider.of<StockTickerProvider>(context, listen: false); // Use listen:false for actions
     final dateFormat = DateFormat('yyyy-MM-dd HH:mm:ss'); // Formatter for timestamp

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4.0),
      child: ListTile(
        title: Text(token, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: stockData != null
            ? Text(
                'Price: ${stockData.stockPrice.toStringAsFixed(2)}\nUpdated: ${dateFormat.format(stockData.processedTimestampUtc.toLocal())}', // Display timestamp in local time
                 style: const TextStyle(fontSize: 12),
              )
            : const Text('Waiting for data...'),
        trailing: IconButton(
          icon: const Icon(Icons.remove_circle_outline, color: Colors.red),
          tooltip: 'Unsubscribe',
          onPressed: () {
            provider.unsubscribe(token);
          },
        ),
      ),
    );
  }
}
