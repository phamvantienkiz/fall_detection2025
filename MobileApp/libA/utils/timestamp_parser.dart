class TimestampParser {
  static Map<String, String> parseTimestamp(String timestamp) {
    if (timestamp.length == 15 && timestamp.contains('_')) {
      try {
        String datePart = timestamp.substring(0, 8);
        String timePart = timestamp.substring(9);

        String year = datePart.substring(0, 4);
        String month = datePart.substring(4, 6);
        String day = datePart.substring(6, 8);

        String hour = timePart.substring(0, 2);
        String minute = timePart.substring(2, 4);
        String second = timePart.substring(4, 6);

        return {
          'date': '$year/$month/$day',
          'time': '$hour:$minute:$second',
        };
      } catch (e) {
        // Return original if parsing fails, or handle error as needed
        return {'date': timestamp, 'time': ''};
      }
    } else {
      // Return original if format is incorrect
      return {'date': timestamp, 'time': ''};
    }
  }
}
