// ignore_for_file: unused_import

import 'package:fall_tracking/services/notification_service.dart';
import 'package:flutter/material.dart';
import 'package:fall_tracking/screen/home_screen.dart';
import 'package:fall_tracking/utils/timestamp_parser.dart';

class FallAlertScreen extends StatefulWidget {
  final Map<String, dynamic> payload;
  const FallAlertScreen({Key? key, required this.payload}) : super(key: key);

  @override
  State<FallAlertScreen> createState() => _FallAlertScreenState();
}

class _FallAlertScreenState extends State<FallAlertScreen> {
  @override
  void initState() {
    super.initState();
    // Hủy tất cả thông báo và dừng rung khi màn hình này được hiển thị
    NotificationService().cancelAllNotifications();
  }

  @override
  Widget build(BuildContext context) {
    // Lấy thông tin từ payload nếu có
    final String event = widget.payload['event']?.toString() ?? 'FALL';
    final String prob = widget.payload['prob']?.toString() ?? '';
    final String? timestamp = widget.payload['timestamp']?.toString();
    final parsedTs =
        timestamp != null ? TimestampParser.parseTimestamp(timestamp) : null;

    return Scaffold(
      // Cho phép body hiển thị đằng sau AppBar
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        // Làm cho AppBar trong suốt để thấy được gradient của body
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white),
          onPressed: () {
            // Xử lý sự kiện khi nhấn nút back, ví dụ: quay lại màn hình trước
            if (Navigator.canPop(context)) {
              Navigator.of(context).pop();
            }
          },
        ),
        title: const Text(
          'Fall Alerts', // Tiêu đề đã được thay đổi
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Container(
        // Đặt nền gradient cho toàn bộ màn hình
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF8A2BE2), Color(0xFF4B0082)], // Dải màu tím
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        // Căn giữa thẻ thông báo
        child: Center(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 24.0),
            padding: const EdgeInsets.all(24.0),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16.0),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.1),
                  blurRadius: 10,
                  spreadRadius: 2,
                ),
              ],
            ),
            // Sử dụng Column để các phần tử bên trong thẻ xếp chồng lên nhau
            child: Column(
              // mainAxisSize.min giúp thẻ co lại vừa với nội dung bên trong
              mainAxisSize: MainAxisSize.min,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Đoạn text "Fall Detected" màu đỏ
                const Text(
                  'Fall Detected',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Color(0xFFF44336), // Màu đỏ đậm
                    fontSize: 36,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 24.0),

                // Icon cảnh báo
                const Icon(
                  Icons.warning_amber_rounded,
                  color: Colors.orangeAccent,
                  size: 80,
                ),
                const SizedBox(height: 24.0),
                if (parsedTs != null) ...[
                  Text(
                    'Date: ${parsedTs['date']}',
                    style: const TextStyle(fontSize: 16, color: Colors.black87),
                  ),
                  Text(
                    'Time: ${parsedTs['time']}',
                    style: const TextStyle(fontSize: 16, color: Colors.black87),
                  ),
                  const SizedBox(height: 8),
                ],
                if (prob.isNotEmpty)
                  Text('Probability: $prob',
                      style:
                          const TextStyle(fontSize: 16, color: Colors.black87)),
                const SizedBox(height: 32.0),
                // Nút "Continue"
                SizedBox(
                  width: double.infinity, // Cho nút rộng hết cỡ thẻ
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.of(context)
                          .pushNamedAndRemoveUntil('/', (route) => false);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor:
                          const Color(0xFF4CAF50), // Màu xanh lá cây
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                      elevation: 5,
                    ),
                    child: const Text(
                      'Continue',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
