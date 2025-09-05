// ignore_for_file: prefer_const_constructors, unused_import

import 'dart:async';

import 'package:fall_tracking/services/api_service.dart'; // Import ApiService
import 'package:fall_tracking/utils/timestamp_parser.dart'; // Import TimestampParser
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:fall_tracking/screen/token_screen.dart';
import 'package:fall_tracking/services/fcm_get_token.dart'; // Import FCMService
import 'package:flutter/foundation.dart'; // for kDebugMode
import 'package:fall_tracking/screen/alert_screen.dart'; // Import AlertScreen
import 'package:fall_tracking/services/notification_service.dart'; // Import NotificationService

class FallTrackingHomeScreen extends StatefulWidget {
  const FallTrackingHomeScreen({super.key});

  @override
  State<FallTrackingHomeScreen> createState() => _FallTrackingHomeScreenState();
}

class _FallTrackingHomeScreenState extends State<FallTrackingHomeScreen> {
  final ApiService _apiService = ApiService();
  List<EventData> _eventList = [];
  bool _isLoading = true;
  String? _errorMessage;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _fetchEvents(isInitial: true);
    _timer = Timer.periodic(const Duration(seconds: 10), (timer) {
      _fetchEvents();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _fetchEvents({bool isInitial = false}) async {
    if (isInitial) {
      setState(() {
        _isLoading = true;
      });
    }
    try {
      final events = await _apiService.fetchData();

      // Sắp xếp các sự kiện theo timestamp giảm dần (mới nhất lên đầu)
      events.sort((a, b) => b.timestamp.compareTo(a.timestamp));

      if (mounted) {
        setState(() {
          _eventList = events;
          _isLoading = false;
          _errorMessage = null;
        });
      }
    } catch (e) {
      if (mounted) {
        // Don't set loading to false here if we want to keep the old data on screen
        // Only update the error message if it's the initial load or the list is empty
        if (isInitial || _eventList.isEmpty) {
          setState(() {
            _errorMessage = e.toString();
            _isLoading = false;
          });
        } else {
          // For subsequent refreshes, show a snackbar or a less intrusive error
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Failed to refresh events: ${e.toString()}'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
      if (kDebugMode) {
        print("Error fetching events in HomeScreen: $e");
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;

    return Scaffold(
      backgroundColor: const Color(0xFFF4F6F8),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : _errorMessage != null
              ? Center(child: Text('Error: $_errorMessage'))
              : RefreshIndicator(
                  // Added RefreshIndicator
                  onRefresh: _fetchEvents,
                  child: SingleChildScrollView(
                    physics:
                        AlwaysScrollableScrollPhysics(), // Ensure scroll even with few items
                    child: Column(
                      children: [
                        Stack(
                          clipBehavior: Clip.none,
                          alignment: Alignment.topCenter,
                          children: [
                            _buildHeader(),
                            Positioned(
                              top: 195, // Pushed down from 150
                              child: _buildDashboardCards(context, screenWidth),
                            ),
                          ],
                        ),
                        const SizedBox(height: 150), // Adjusted spacing
                        _buildUpcomingList(),
                      ],
                    ),
                  ),
                ),
    );
  }

  // Widget cho phần Header màu tím
  Widget _buildHeader() {
    return Container(
      height: 230,
      padding: const EdgeInsets.fromLTRB(20, 50, 20, 20),
      decoration: const BoxDecoration(
        color: Color(0xFF6A1B9A),
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(40),
          bottomRight: Radius.circular(40),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: const [
                  Icon(Icons.crisis_alert, color: Colors.white, size: 28),
                  SizedBox(width: 10),
                  Text(
                    'FALL TRACKING',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(Icons.qr_code_scanner_outlined,
                    color: Colors.white),
              ),
            ],
          ),
          const SizedBox(height: 20),
          const Text(
            'Welcome to Fall Tracking!',
            style: TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const Text(
            'Stay safe and monitor fall events.',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }

  // Widget cho hai thẻ "Upcoming" và "Create Token"
  Widget _buildDashboardCards(BuildContext context, double screenWidth) {
    final FCMService fcmService = FCMService(); // Create instance of FCMService

    return Container(
      width: screenWidth,
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          Expanded(
            // Wrap with Expanded
            child: GestureDetector(
              behavior: HitTestBehavior.opaque, // Make the whole area tappable
              onTap: () {
                Navigator.pushNamed(context, '/listFall');
              },
              child: _DashboardCard(
                color: const Color(0xFFA6FFD7),
                icon: Icons.calendar_today_outlined,
                title: 'Event',
                //subtitle: _eventList.length.toString(), // Show number of events
                textColor: const Color(0xFF00BFA5),
                iconColor: const Color(0xFF00BFA5),
              ),
            ),
          ),
          const SizedBox(width: 20),
          Expanded(
            // Wrap with Expanded
            child: GestureDetector(
              behavior: HitTestBehavior.opaque, // Make the whole area tappable
              onTap: () async {
                // Make onTap async
                String? token = await fcmService.getFCMToken();
                if (kDebugMode) {
                  print("Token from home_screen: $token");
                }
                if (context.mounted) {
                  // Check if context is still valid
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => TokenDetailsScreen(
                              tokenCode:
                                  token ?? 'N/A', // Pass the token or 'N/A'
                            )),
                  );
                }
              },
              child: _DashboardCard(
                color: const Color(0xFFD6C8FF),
                icon: Icons.add,
                title: 'Create Token',
                textColor: const Color(0xFF6A1B9A),
                iconColor: const Color(0xFF6A1B9A),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // Widget cho phần danh sách "Upcoming List"
  Widget _buildUpcomingList() {
    if (_eventList.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Text(
            'No events found.',
            style: TextStyle(fontSize: 16, color: Colors.grey),
          ),
        ),
      );
    }
    // Take only the first 3 events
    final displayedEvents = _eventList.take(3).toList();

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20.0),
      child: Column(
        children: [
          // Tiêu đề danh sách
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: const [
              Text(
                'Event List', // Changed title
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
              ),
              // Đã xóa Container hiển thị số lượng sự kiện ở đây
            ],
          ),
          const SizedBox(height: 20),

          // Các mục trong danh sách
          ListView.builder(
            // Use ListView.builder
            shrinkWrap: true, // Important for ListView inside Column
            physics:
                NeverScrollableScrollPhysics(), // Disable scrolling for this ListView
            itemCount: displayedEvents.length, // Use the limited list
            itemBuilder: (context, index) {
              final event = displayedEvents[index]; // Use the limited list
              final parsedTs = TimestampParser.parseTimestamp(event.timestamp);
              return Padding(
                padding: const EdgeInsets.only(bottom: 15.0),
                child: _QueueListItem(
                  event: event.event, // Changed from tokenCode
                  prob: event.prob.toStringAsFixed(
                      4), // Changed from counterInfo, formatted prob
                  date: parsedTs['date']!,
                  timeRange: parsedTs['time']!,
                  hasBorder: index == 0, // Example: highlight first item
                ),
              );
            },
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }
}

// Widget tái sử dụng cho các thẻ trên dashboard
class _DashboardCard extends StatelessWidget {
  final Color color;
  final IconData icon;
  final String title;
  final String? subtitle;
  final Color textColor;
  final Color iconColor;

  const _DashboardCard({
    required this.color,
    required this.icon,
    required this.title,
    this.subtitle,
    required this.textColor,
    required this.iconColor,
  });

  @override
  Widget build(BuildContext context) {
    // Bỏ Expanded ở đây vì nó đã được dùng bên ngoài khi gọi widget này.
    return Container(
      height: 130,
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            spreadRadius: 2,
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white,
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: iconColor, size: 28),
          ),
          const SizedBox(height: 4),
          Text(
            title,
            style: TextStyle(
              color: Colors.black87,
              fontWeight: FontWeight.w600,
            ),
          ),
          if (subtitle != null)
            Text(
              subtitle!,
              style: TextStyle(
                color: textColor,
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
        ],
      ),
    );
  }
}

// Widget tái sử dụng cho mỗi mục trong danh sách hàng đợi
class _QueueListItem extends StatelessWidget {
  final String event; // Changed from tokenCode
  final String prob; // Changed from counterInfo
  final String date;
  final String timeRange;
  final bool hasBorder;

  const _QueueListItem({
    required this.event,
    required this.prob,
    required this.date,
    required this.timeRange,
    this.hasBorder = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        border: hasBorder
            ? Border.all(color: const Color(0xFF00BFA5), width: 1.5)
            : null,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          // Thông tin token và quầy
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Event: $event', // Display event
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Prob: $prob', // Display probability
                  style: const TextStyle(
                    color: Colors.black54,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),
          // Thông tin thời gian
          Row(
            children: [
              const Icon(Icons.access_time, color: Color(0xFF6A1B9A), size: 20),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    date,
                    style: const TextStyle(
                      color: Colors.black54,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    timeRange,
                    style: const TextStyle(
                      color: Color(0xFF6A1B9A),
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ],
          )
        ],
      ),
    );
  }
}
