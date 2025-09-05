import 'dart:async';

import 'package:fall_tracking/services/api_service.dart'; // Import ApiService
import 'package:fall_tracking/utils/timestamp_parser.dart'; // Import TimestampParser
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart'; // for kDebugMode

// Lớp để lưu trữ thông tin của một sự kiện (thay thế TicketInfo)
class EventDisplayInfo {
  final String event; // Formerly ticketNo
  final String prob; // Formerly counterNo, now stores probability as string
  final String date;
  final String time;

  EventDisplayInfo({
    required this.event,
    required this.prob,
    required this.date,
    required this.time,
  });
}

class ListFallScreen extends StatefulWidget {
  const ListFallScreen({super.key});

  @override
  State<ListFallScreen> createState() => _ListFallScreenState();
}

class _ListFallScreenState extends State<ListFallScreen> {
  final ApiService _apiService = ApiService();
  List<EventDisplayInfo> _eventDisplayList = [];
  bool _isLoading = true;
  String? _errorMessage;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _fetchAndPrepareEvents(isInitial: true);
    _timer = Timer.periodic(const Duration(seconds: 10), (timer) {
      _fetchAndPrepareEvents();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _fetchAndPrepareEvents({bool isInitial = false}) async {
    if (isInitial) {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
      });
    }
    try {
      final eventsFromApi = await _apiService.fetchData();

      // Sắp xếp các sự kiện theo timestamp giảm dần (mới nhất lên đầu)
      eventsFromApi.sort((a, b) => b.timestamp.compareTo(a.timestamp));

      final displayList = eventsFromApi.map((eventData) {
        final parsedTs = TimestampParser.parseTimestamp(eventData.timestamp);
        return EventDisplayInfo(
          event: eventData.event,
          prob: eventData.prob.toStringAsFixed(4),
          date: parsedTs['date']!,
          time: parsedTs['time']!,
        );
      }).toList();
      if (mounted) {
        setState(() {
          _eventDisplayList = displayList;
          _isLoading = false;
          _errorMessage = null;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          if (isInitial || _eventDisplayList.isEmpty) {
            _errorMessage = e.toString();
          }
          _isLoading = false;
        });
      }
      if (kDebugMode) {
        print("Error fetching events in UpcomingListScreen: $e");
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white),
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
        title: const Text(
          'Fall Event List',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFF8A2BE2), Color(0xFF6A1B9A)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
        ),
      ),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : _errorMessage != null
              ? Center(
                  child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text('Error: $_errorMessage\nPull down to refresh.',
                      textAlign: TextAlign.center),
                ))
              : RefreshIndicator(
                  onRefresh: _fetchAndPrepareEvents,
                  child: _eventDisplayList.isEmpty
                      ? Center(
                          child: Text("No events found. Pull down to refresh.",
                              style: TextStyle(color: Colors.grey[600])),
                        )
                      : ListView.builder(
                          padding: const EdgeInsets.all(16.0),
                          itemCount: _eventDisplayList.length,
                          itemBuilder: (context, index) {
                            final eventItem = _eventDisplayList[index];
                            return EventCard(
                              eventInfo: eventItem,
                            );
                          },
                        ),
                ),
    );
  }
}

class EventCard extends StatelessWidget {
  final EventDisplayInfo eventInfo;

  const EventCard({
    super.key,
    required this.eventInfo,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16.0),
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(12.0),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 5,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        children: [
          // Bỏ số thứ tự
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Event: ${eventInfo.event}',
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16.0,
                  ),
                ),
                const SizedBox(height: 4.0),
                Text(
                  'Prob: ${eventInfo.prob}',
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 14.0,
                  ),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              const Icon(
                Icons.watch_later_outlined,
                color: Color(0xFF6A1B9A),
                size: 20,
              ),
              const SizedBox(height: 4),
              Text(
                eventInfo.date,
                style: const TextStyle(
                    color: Color(0xFF6A1B9A),
                    fontSize: 12.0,
                    fontWeight: FontWeight.w500),
              ),
              Text(
                'At ${eventInfo.time}',
                style: const TextStyle(
                    color: Color(0xFF6A1B9A),
                    fontSize: 12.0,
                    fontWeight: FontWeight.w500),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
