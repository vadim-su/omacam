import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Commons as Commons
import qs.Ui

BarWidget {
  id: root
  moduleName: "vadim-su.omacam"
  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  property bool popupOpen: false
  property bool daemonConnected: false
  property bool cameraStreaming: false
  property bool effectsActive: true
  property string bgMode: "blur"
  property real blurStrength: 0.65
  property real studioLight: 0.45
  property real sharpness: 0.65
  property bool centerStage: false
  property real centerStageZoom: 0.55
  property real bgImageBlur: 0.15
  property string bgImagePath: ""
  property string bgColorHex: "#1c2433"

  readonly property string omacamBin: (Quickshell.env("HOME") || "") + "/.local/bin/omacam"
  readonly property string pickBgScript: Qt.resolvedUrl("pick-bg.sh").toString().replace(/^file:\/\//, "")

  readonly property var presetColors: [
    "#1c2433",
    "#12151f",
    "#282422",
    "#182924",
    "#241c35",
    "#d8d2c7"
  ]

  function close() {
    popupOpen = false
  }

  function basenameOf(p) {
    if (!p || p.length === 0) return ""
    var parts = p.split("/")
    var name = parts[parts.length - 1]
    if (name.length > 36) return name.slice(0, 33) + "…"
    return name
  }

  function formatPct(v) {
    return Commons.I18n.formatNumber(Math.round(Math.max(0, Math.min(1, v)) * 100)) + "%"
  }

  function sendCommand(obj) {
    if (!watcher.running) return
    watcher.write(JSON.stringify(obj) + "\n")
  }

  function sendPatch(patch, persist) {
    sendCommand({
      cmd: "set",
      patch: patch,
      persist: persist !== false
    })
  }

  function applySnapshot(line) {
    if (!line || line.length === 0) return
    try {
      var s = JSON.parse(line)
      if (typeof s.connected === "boolean") root.daemonConnected = s.connected
      if (typeof s.streaming === "boolean") root.cameraStreaming = s.streaming
      if (typeof s.enabled === "boolean") root.effectsActive = s.enabled
      if (typeof s.background_mode === "string") root.bgMode = s.background_mode
      if (typeof s.blur_strength === "number") root.blurStrength = s.blur_strength
      if (typeof s.studio_light === "number") root.studioLight = s.studio_light
      if (typeof s.sharpness === "number") root.sharpness = s.sharpness
      if (typeof s.center_stage === "boolean") root.centerStage = s.center_stage
      if (typeof s.center_stage_zoom === "number") root.centerStageZoom = s.center_stage_zoom
      if (typeof s.bg_image_blur === "number") root.bgImageBlur = s.bg_image_blur
      root.bgImagePath = (typeof s.bg_image_path === "string") ? s.bg_image_path : ""
      if (typeof s.bg_color_hex === "string") root.bgColorHex = s.bg_color_hex
    } catch (e) {
      // Ignore malformed IPC lines
    }
  }

  Commons.Translations {
    id: strings
    directory: Qt.resolvedUrl("i18n")
  }

  Process {
    id: watcher
    running: true
    stdinEnabled: true
    command: [root.omacamBin, "ctl", "watch"]
    stdout: SplitParser {
      onRead: function(line) { root.applySnapshot(line) }
    }
    onExited: function() {
      root.daemonConnected = false
      root.cameraStreaming = false
    }
  }

  Process {
    id: imagePickerProc
    running: false
    command: ["/bin/bash", root.pickBgScript]
  }

  WidgetButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: (root.daemonConnected && root.effectsActive) ? "󰕧" : "󰕨"
    active: root.cameraStreaming
    tooltipText: !root.daemonConnected
      ? strings.t("tooltip.offline")
      : (root.cameraStreaming ? strings.t("tooltip.live") : strings.t("tooltip.idle"))
    onPressed: function(mouseButton) {
      if (mouseButton === Qt.RightButton) {
        if (root.daemonConnected) {
          root.effectsActive = !root.effectsActive
          root.sendCommand({ cmd: "toggle" })
        }
      } else {
        root.popupOpen = !root.popupOpen
      }
    }
  }

  PopupCard {
    id: popup
    anchorItem: root
    bar: root.bar
    owner: root
    open: root.popupOpen
    contentWidth: popup.fittedContentWidth(Style.space(316))
    contentHeight: popup.fittedContentHeight(contentColumn.implicitHeight)

    Column {
      id: contentColumn
      width: parent.width
      spacing: Style.space(10)

      // Header row: title + live status pill + master switch
      Row {
        width: parent.width
        spacing: Style.space(8)

        Text {
          textFormat: Text.PlainText
          text: "󰕧"
          color: root.cameraStreaming ? Commons.Color.accent : Commons.Color.popups.text
          font.family: Style.font.family
          font.pixelSize: Style.font.subtitle
          anchors.verticalCenter: parent.verticalCenter
        }

        Column {
          width: parent.width - masterSwitch.implicitWidth - Style.space(34)
          spacing: Style.space(1)
          anchors.verticalCenter: parent.verticalCenter

          Text {
            textFormat: Text.PlainText
            text: strings.t("title")
            color: Commons.Color.popups.text
            font.family: Style.font.family
            font.pixelSize: Style.font.subtitle
            font.bold: true
            elide: Text.ElideRight
            width: parent.width
          }

          Row {
            spacing: Style.space(6)
            Rectangle {
              width: Style.space(7)
              height: Style.space(7)
              radius: width / 2
              anchors.verticalCenter: parent.verticalCenter
              color: !root.daemonConnected
                ? Qt.darker(Commons.Color.popups.text, 2.0)
                : (root.cameraStreaming ? Commons.Color.accent : Qt.darker(Commons.Color.popups.text, 1.4))
            }
            Text {
              textFormat: Text.PlainText
              text: !root.daemonConnected
                ? strings.t("status.offline")
                : (root.cameraStreaming ? strings.t("status.live") : strings.t("status.idle"))
              color: root.cameraStreaming ? Commons.Color.accent : Qt.darker(Commons.Color.popups.text, 1.35)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
              font.bold: root.cameraStreaming
            }
          }
        }

        ToggleSwitch {
          id: masterSwitch
          anchors.verticalCenter: parent.verticalCenter
          checked: root.daemonConnected && root.effectsActive
          busy: !root.daemonConnected
          foreground: Commons.Color.popups.text
          accent: Commons.Color.accent
          onToggled: {
            if (!root.daemonConnected) return
            root.effectsActive = !root.effectsActive
            root.sendPatch({ enabled: root.effectsActive }, true)
          }
        }
      }

      // Offline hint
      Text {
        visible: !root.daemonConnected
        width: parent.width
        wrapMode: Text.WordWrap
        textFormat: Text.PlainText
        text: strings.t("offline.hint")
        color: Qt.darker(Commons.Color.popups.text, 1.3)
        font.family: Style.font.family
        font.pixelSize: Style.font.caption
      }

      // Controls (shown when daemon is connected)
      Column {
        visible: root.daemonConnected
        width: parent.width
        spacing: Style.space(10)
        opacity: root.effectsActive ? 1.0 : 0.45

        Behavior on opacity {
          NumberAnimation { duration: Style.duration(140); easing.type: Easing.OutCubic }
        }

        PanelSeparator {
          width: parent.width
        }

        // Center Stage toggle + framing zoom slider (prominent at top!)
        Row {
          width: parent.width
          spacing: Style.space(8)

          Text {
            textFormat: Text.PlainText
            text: strings.t("center_stage.title")
            color: Commons.Color.popups.text
            font.family: Style.font.family
            font.pixelSize: Style.font.body
            font.bold: true
            width: parent.width - csSwitch.implicitWidth - Style.space(8)
            anchors.verticalCenter: parent.verticalCenter
            elide: Text.ElideRight
          }

          ToggleSwitch {
            id: csSwitch
            anchors.verticalCenter: parent.verticalCenter
            checked: root.centerStage
            foreground: Commons.Color.popups.text
            accent: Commons.Color.accent
            onToggled: {
              root.centerStage = !root.centerStage
              root.effectsActive = true
              root.sendPatch({ enabled: true, center_stage: root.centerStage }, true)
            }
          }
        }

        Column {
          visible: root.centerStage
          width: parent.width
          spacing: Style.space(2)

          Item {
            width: parent.width
            height: csZoomLabel.implicitHeight
            Text {
              id: csZoomLabel
              textFormat: Text.PlainText
              text: strings.t("center_stage.zoom")
              color: Commons.Color.popups.text
              font.family: Style.font.family
              font.pixelSize: Style.font.body
              anchors.left: parent.left
            }
            Text {
              textFormat: Text.PlainText
              text: root.formatPct(csZoomSlider.liveValue)
              color: Qt.darker(Commons.Color.popups.text, 1.3)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
              anchors.right: parent.right
              anchors.verticalCenter: csZoomLabel.verticalCenter
            }
          }

          PanelSlider {
            id: csZoomSlider
            width: parent.width
            bar: root.bar
            value: root.centerStageZoom
            minimum: 0
            maximum: 1
            step: 0.05
            tickCount: 5
            onMoved: function(v) {
              root.centerStageZoom = v
              root.sendPatch({ center_stage_zoom: v }, false)
            }
            onReleased: function(v) {
              root.centerStageZoom = v
              root.sendPatch({ center_stage_zoom: v }, true)
            }
          }
        }

        PanelSeparator {
          width: parent.width
        }

        // Background mode selector
        PanelSectionHeader {
          text: strings.t("section.background")
          foreground: Commons.Color.popups.text
        }

        ButtonGroup {
          id: bgModeGroup
          width: parent.width
          value: root.bgMode
          foreground: Commons.Color.popups.text
          background: Commons.Color.popups.background
          accent: Commons.Color.accent
          focusable: false
          options: [
            { value: "none",  label: strings.t("mode.none") },
            { value: "blur",  label: strings.t("mode.blur") },
            { value: "color", label: strings.t("mode.color") },
            { value: "image", label: strings.t("mode.image") }
          ]
          onChanged: function(val) {
            root.bgMode = val
            root.effectsActive = true
            root.sendPatch({ enabled: true, background_mode: val }, true)
          }
        }

        // Portrait Blur slider (when mode == blur)
        Column {
          visible: root.bgMode === "blur"
          width: parent.width
          spacing: Style.space(2)

          Item {
            width: parent.width
            height: blurLabel.implicitHeight
            Text {
              id: blurLabel
              textFormat: Text.PlainText
              text: strings.t("slider.blur")
              color: Commons.Color.popups.text
              font.family: Style.font.family
              font.pixelSize: Style.font.body
              anchors.left: parent.left
            }
            Text {
              textFormat: Text.PlainText
              text: root.formatPct(blurSlider.liveValue)
              color: Qt.darker(Commons.Color.popups.text, 1.3)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
              anchors.right: parent.right
              anchors.verticalCenter: blurLabel.verticalCenter
            }
          }

          PanelSlider {
            id: blurSlider
            width: parent.width
            bar: root.bar
            value: root.blurStrength
            minimum: 0
            maximum: 1
            step: 0.05
            tickCount: 5
            onMoved: function(v) {
              root.blurStrength = v
              root.sendPatch({ blur_strength: v }, false)
            }
            onReleased: function(v) {
              root.blurStrength = v
              root.sendPatch({ blur_strength: v }, true)
            }
          }
        }

        // Color swatches (when mode == color)
        Column {
          visible: root.bgMode === "color"
          width: parent.width
          spacing: Style.space(6)

          PanelSectionHeader {
            text: strings.t("section.backdrop_color")
            foreground: Commons.Color.popups.text
          }

          Row {
            spacing: Style.space(8)
            Repeater {
              model: root.presetColors
              Rectangle {
                required property string modelData
                readonly property bool isPicked: root.bgColorHex.toLowerCase() === modelData.toLowerCase()
                width: Style.space(28)
                height: Style.space(28)
                radius: Style.cornerRadius > 0 ? Style.space(7) : 0
                color: modelData
                border.width: isPicked ? 2 : 1
                border.color: isPicked ? Commons.Color.accent : Qt.rgba(1, 1, 1, 0.2)

                MouseArea {
                  anchors.fill: parent
                  cursorShape: Qt.PointingHandCursor
                  onClicked: {
                    root.bgColorHex = parent.modelData
                    root.sendPatch({ background_mode: "color", bg_color_hex: parent.modelData }, true)
                  }
                }
              }
            }
          }
        }

        // Image picker controls (when mode == image)
        Column {
          visible: root.bgMode === "image"
          width: parent.width
          spacing: Style.space(6)

          PanelSectionHeader {
            text: strings.t("section.backdrop_image")
            foreground: Commons.Color.popups.text
          }

          Text {
            width: parent.width
            textFormat: Text.PlainText
            text: root.bgImagePath.length > 0 ? root.basenameOf(root.bgImagePath) : strings.t("image.none")
            color: Qt.darker(Commons.Color.popups.text, 1.25)
            font.family: Style.font.family
            font.pixelSize: Style.font.caption
            elide: Text.ElideMiddle
          }

          Row {
            spacing: Style.space(8)
            Button {
              text: strings.t("image.pick")
              bordered: true
              foreground: Commons.Color.popups.text
              accent: Commons.Color.accent
              onClicked: {
                root.popupOpen = false
                if (!imagePickerProc.running) imagePickerProc.running = true
              }
            }
            Button {
              visible: root.bgImagePath.length > 0
              text: strings.t("image.clear")
              bordered: true
              foreground: Commons.Color.popups.text
              accent: Commons.Color.accent
              onClicked: {
                root.bgImagePath = ""
                root.sendPatch({ bg_image_path: null }, true)
              }
            }
          }

          Item {
            width: parent.width
            height: bgBlurLabel.implicitHeight
            Text {
              id: bgBlurLabel
              textFormat: Text.PlainText
              text: strings.t("slider.bg_blur")
              color: Commons.Color.popups.text
              font.family: Style.font.family
              font.pixelSize: Style.font.body
              anchors.left: parent.left
            }
            Text {
              textFormat: Text.PlainText
              text: root.formatPct(bgBlurSlider.liveValue)
              color: Qt.darker(Commons.Color.popups.text, 1.3)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
              anchors.right: parent.right
              anchors.verticalCenter: bgBlurLabel.verticalCenter
            }
          }

          PanelSlider {
            id: bgBlurSlider
            width: parent.width
            bar: root.bar
            value: root.bgImageBlur
            minimum: 0
            maximum: 1
            step: 0.05
            tickCount: 5
            onMoved: function(v) {
              root.bgImageBlur = v
              root.sendPatch({ bg_image_blur: v }, false)
            }
            onReleased: function(v) {
              root.bgImageBlur = v
              root.sendPatch({ bg_image_blur: v }, true)
            }
          }
        }

        PanelSeparator {
          width: parent.width
        }

        // Studio Light slider
        Column {
          width: parent.width
          spacing: Style.space(2)

          Item {
            width: parent.width
            height: slLabel.implicitHeight
            Text {
              id: slLabel
              textFormat: Text.PlainText
              text: strings.t("slider.studio_light")
              color: Commons.Color.popups.text
              font.family: Style.font.family
              font.pixelSize: Style.font.body
              anchors.left: parent.left
            }
            Text {
              textFormat: Text.PlainText
              text: root.formatPct(slSlider.liveValue)
              color: Qt.darker(Commons.Color.popups.text, 1.3)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
              anchors.right: parent.right
              anchors.verticalCenter: slLabel.verticalCenter
            }
          }

          PanelSlider {
            id: slSlider
            width: parent.width
            bar: root.bar
            value: root.studioLight
            minimum: 0
            maximum: 1
            step: 0.05
            tickCount: 5
            onMoved: function(v) {
              root.studioLight = v
              root.sendPatch({ studio_light: v }, false)
            }
            onReleased: function(v) {
              root.studioLight = v
              root.sendPatch({ studio_light: v }, true)
            }
          }
        }

        // Sharpness & Clarity slider (fixes soft apple-isp look!)
        Column {
          width: parent.width
          spacing: Style.space(2)

          Item {
            width: parent.width
            height: sharpLabel.implicitHeight
            Text {
              id: sharpLabel
              textFormat: Text.PlainText
              text: strings.t("slider.sharpness")
              color: Commons.Color.popups.text
              font.family: Style.font.family
              font.pixelSize: Style.font.body
              anchors.left: parent.left
            }
            Text {
              textFormat: Text.PlainText
              text: root.formatPct(sharpSlider.liveValue)
              color: Qt.darker(Commons.Color.popups.text, 1.3)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
              anchors.right: parent.right
              anchors.verticalCenter: sharpLabel.verticalCenter
            }
          }

          PanelSlider {
            id: sharpSlider
            width: parent.width
            bar: root.bar
            value: root.sharpness
            minimum: 0
            maximum: 1
            step: 0.05
            tickCount: 5
            onMoved: function(v) {
              root.sharpness = v
              root.sendPatch({ sharpness: v }, false)
            }
            onReleased: function(v) {
              root.sharpness = v
              root.sendPatch({ sharpness: v }, true)
            }
          }
        }
      }
    }
  }
}
