import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart'; // For kDebugMode

// Data model for the API response
class EventData {
  final String timestamp;
  final String event;
  final double prob;

  EventData({
    required this.timestamp,
    required this.event,
    required this.prob,
  });

  factory EventData.fromJson(Map<String, dynamic> json) {
    return EventData(
      timestamp: json['timestamp'] as String,
      event: json['event'] as String,
      prob: (json['prob'] as num).toDouble(),
    );
  }

  @override
  String toString() {
    return 'EventData{timestamp: $timestamp, event: $event, prob: $prob}';
  }
}

class ApiService {
  static const String _baseUrl =
      'https://your-api-endpoint.com/api/events'; // Replace with your actual API endpoint

  Future<List<EventData>> fetchData() async {
    try {
      final response = await http.get(Uri.parse(_baseUrl));

      if (response.statusCode == 200) {
        // If the server returns a 200 OK response,
        // then parse the JSON.
        List<dynamic> body = jsonDecode(response.body);
        List<EventData> data = body
            .map(
              (dynamic item) =>
                  EventData.fromJson(item as Map<String, dynamic>),
            )
            .toList();
        if (kDebugMode) {
          print('Fetched data: $data');
        }
        return data;
      } else {
        // If the server did not return a 200 OK response,
        // then throw an exception.
        if (kDebugMode) {
          print('Failed to load data. Status code: \${response.statusCode}');
          print('Response body: \${response.body}');
        }
        throw Exception(
            'Failed to load data from API. Status code: \${response.statusCode}');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error fetching data: $e');
      }
      // Rethrow the exception to be handled by the caller
      throw Exception('Error fetching data: $e');
    }
  }
}
