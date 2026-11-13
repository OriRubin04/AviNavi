using System;
using System.Windows.Forms;
using MissionPlanner;
using MissionPlanner.Plugin;
using MissionPlanner.Utilities;

// AviNavi: adds "AviNavi: Search Here" to the Flight Data map's right-click
// menu. Clicking it tells the companion computer where to resume searching for
// a visual position fix, without touching anything on the aircraft.
//
// Why a plugin rather than a stock menu item:
//
//   Every built-in map action is addressed to target_component = 1 (the
//   autopilot). ArduPilot's MAVLink_routing::forward hits an early return for
//   those and never forwards them to other links, so a companion computer on a
//   different serial port cannot see them. This plugin addresses the command to
//   the companion directly, which both solves that and makes the intent
//   explicit.
//
//   The command is MAV_CMD_SPATIAL_USER_1 (31005) - the slot the MAVLink
//   specification reserves for a user-defined point on the ground. It is not in
//   ArduPilot's command_long_stores_location(), so if the autopilot receives a
//   copy it answers MAV_RESULT_UNSUPPORTED and does nothing. No flight mode
//   change, no home change, no RTL change, no EKF origin change.
//
//   COMMAND_INT is used rather than COMMAND_LONG because its x/y are degE7
//   integers (about 1 cm of resolution) while COMMAND_LONG's param5/param6 are
//   float32 degrees (about 1 m near the poles).
//
// Build:
//   csc /target:library /out:AviNaviSearchHere.dll /r:MissionPlanner.exe \
//       /r:MissionPlanner.Utilities.dll /r:MAVLink.dll AviNaviSearchHere.cs
//   ...then drop the DLL into Mission Planner's plugins\ directory and restart.
//
// On the Pi, set in config/default.yaml:
//   reanchor:
//     accept_spatial_user: true
//   mavlink:
//     component_id: 191        # MAV_COMP_ID_ONBOARD_COMPUTER
//
// IMPORTANT for the "Pi on its own flight-controller UART" wiring: the
// autopilot only forwards to a link it has *learned* a route for, which it does
// from the companion's own traffic. AviNavi sends a continuous heartbeat for
// exactly this reason - leave mavlink.send_heartbeat at true.

namespace AviNavi
{
    public class SearchHerePlugin : MissionPlanner.Plugin.Plugin
    {
        // MAV_CMD_SPATIAL_USER_1. "User defined spatial item ... Example: ROI item."
        private const ushort MAV_CMD_SPATIAL_USER_1 = 31005;

        // MAV_COMP_ID_ONBOARD_COMPUTER. Must match mavlink.component_id on the Pi.
        private const byte COMPANION_COMPONENT_ID = 191;

        // Search radius handed to the companion, in metres, via param1.
        private const float DEFAULT_SEARCH_RADIUS_M = 1500.0f;

        public override string Name { get { return "AviNavi Search Here"; } }
        public override string Version { get { return "1.0"; } }
        public override string Author { get { return "AviNavi"; } }

        public override bool Init()
        {
            loopratehz = 0; // event-driven only; no periodic work
            return true;
        }

        public override bool Loaded()
        {
            var item = new ToolStripMenuItem("AviNavi: Search Here");
            item.Click += OnSearchHereClicked;
            Host.FDMenuMap.Items.Add(item);
            return true;
        }

        private void OnSearchHereClicked(object sender, EventArgs e)
        {
            // Where the right-click that opened the menu landed.
            var point = Host.FDMenuMapPosition;

            if (point.Lat == 0.0 && point.Lng == 0.0)
            {
                CustomMessageBox.Show(
                    "No map position for that click.", "AviNavi");
                return;
            }

            float radius = DEFAULT_SEARCH_RADIUS_M;
            string answer = "";
            if (InputBox.Show(
                    "Search radius",
                    "Radius in metres for the companion's tile search:",
                    ref answer) == DialogResult.OK)
            {
                float parsed;
                if (float.TryParse(answer, out parsed) && parsed > 0)
                {
                    radius = parsed;
                }
            }

            try
            {
                // x and y are degE7 integers. Do not pass float degrees here.
                int latE7 = (int)Math.Round(point.Lat * 1e7);
                int lonE7 = (int)Math.Round(point.Lng * 1e7);

                bool accepted = MainV2.comPort.doCommandInt(
                    MainV2.comPort.sysidcurrent,  // same system as the autopilot
                    COMPANION_COMPONENT_ID,       // but addressed to the companion
                    (MAVLink.MAV_CMD)MAV_CMD_SPATIAL_USER_1,
                    radius,   // param1: search radius, metres (AviNavi convention)
                    0, 0, 0,  // param2..param4: unused
                    latE7,    // x: latitude,  degE7
                    lonE7,    // y: longitude, degE7
                    0,        // z: altitude, metres (unused by AviNavi)
                    (byte)MAVLink.MAV_FRAME.GLOBAL);

                if (accepted)
                {
                    // The companion ACKed, so we know it actually arrived.
                    Host.MainForm.Invoke((MethodInvoker)delegate
                    {
                        CustomMessageBox.Show(
                            string.Format(
                                "Companion will search at\n{0:F7}, {1:F7}\nradius {2:F0} m",
                                point.Lat, point.Lng, radius),
                            "AviNavi");
                    });
                }
                else
                {
                    // No ACK. Either AviNavi is not running, reanchor is
                    // disabled, the component ID does not match, or - in the
                    // separate-UART wiring - the autopilot has not learned a
                    // route to the companion yet.
                    CustomMessageBox.Show(
                        "No acknowledgement from the companion computer.\n\n" +
                        "Check that AviNavi is running, that reanchor.accept_spatial_user " +
                        "is true, and that mavlink.component_id is " +
                        COMPANION_COMPONENT_ID + ".",
                        "AviNavi");
                }
            }
            catch (Exception ex)
            {
                CustomMessageBox.Show("Send failed: " + ex.Message, "AviNavi");
            }
        }
    }
}
