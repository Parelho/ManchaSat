#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "aux_functions.h"

struct can_frame_mauasat {
  int can_id;
  char data[8];
};

int main()
{
  struct can_frame_mauasat *frames = malloc(sizeof(struct can_frame_mauasat) * 1000);
  int frames_pos = 0;

  time_t time_last, time_now;
  time(&time_last);

  while (1) {
    int can_id = 0;
    char data[8] = {0};
    int result = recieve_can(&can_id, data);

    if (!result) {
      // Check if a frame for this can_id was already accounted for during the last second
      int frame_exists = 0;
      int pos_to_replace = -1;
      for (int i = 0; i < frames_pos; i++) {
        if (frames[i].can_id == can_id) {
          frame_exists = 1;
          pos_to_replace = i;
          // printf("Replacing frame\n");
          break;
        }
      }

      // Add/Update frames with latest can
      struct can_frame_mauasat new_frame;
      new_frame.can_id = can_id;
      for (int i = 0; i < 8; i++) {
        new_frame.data[i] = data[i];
      }
      if (frame_exists) {
        frames[pos_to_replace] = new_frame;
      } else {
        frames[frames_pos++] = new_frame;
      }

      // Check if one second has passed to send UART
      time(&time_now);
      if (time_now >= time_last + 1) {
        printf("Recieved cans:\n");
        for (int i = 0; i < frames_pos; i++) {
          switch (frames[i].can_id) {
            case 0: {
              int lat = frames[i].data[0];
              int lon = frames[i].data[1];
              int mmsi = (frames[i].data[2]) | (frames[i].data[3] << 8) | (frames[i].data[4] << 16) | (frames[i].data[5] << 24);

              printf("Recieved AIS: lat: %d, lon: %d, mmsi:%X\n", lat, lon, mmsi);
              break;
            }
            default: {
              printf("Recieved can not wanted\n");
            }
          }
        }

        frames_pos = 0; // Resets frames index
        memset(frames, 0, sizeof(frames)*1000); // Clear old values from frames
        time(&time_last);
      }
    }
  }

  free(frames);
  return 0;
}