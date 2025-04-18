// 디버깅을 위한 콘솔 로그 추가
console.log("DOM 로딩 시작");

// 전역 변수
let speechConfig;
let avatarSynthesizer;
let peerConnection;
let isSpeaking = false;
let sessionActive = false;
let ws;

// WebSocket 연결 설정
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${window.location.host}/api/ws`);

  ws.onopen = function () {
    console.log("WebSocket 연결 성공");
  };

  ws.onmessage = function (event) {
    const data = JSON.parse(event.data);
    console.log("WebSocket 메시지 수신:", data);

    if (data.error) {
      addMessage("system", data.error);
    } else {
      if (data.text_response) {
        addMessage("assistant-text", data.text_response);
      }

      if (data.talk_response) {
        if (sessionActive && avatarSynthesizer) {
          speak(data.talk_response);
        }
      }
    }
    document.getElementById("loader").style.display = "none";
    document.getElementById("sendButton").disabled = false;
  };

  ws.onerror = function (error) {
    console.error("WebSocket 오류:", error);
    addMessage(
      "system",
      "API 서버 연결 오류가 발생했습니다. 잠시 후 자동으로 재연결을 시도합니다."
    );
    // 재연결 시도 시간을 늘림
    setTimeout(connectWebSocket, 5000);
  };

  ws.onclose = function () {
    console.log("WebSocket 연결 종료");
    setTimeout(connectWebSocket, 5000); // 5초 후 재연결 시도
  };
}

// 페이지 로드 시 WebSocket 연결
window.onload = connectWebSocket;

// 단일 DOMContentLoaded 이벤트 리스너
document.addEventListener("DOMContentLoaded", () => {
  console.log("DOMContentLoaded 이벤트 발생");

  // 모든 버튼 요소 가져오기
  const sendButton = document.getElementById("sendButton");
  const startSessionButton = document.getElementById("startSessionButton");
  const stopSessionButton = document.getElementById("stopSessionButton");
  const stopSpeakingButton = document.getElementById("stopSpeakingButton");
  const clearChatButton = document.getElementById("clearChatButton");

  // 디버깅을 위한 로그
  console.log("버튼 요소:", {
    sendButton,
    startSessionButton,
    stopSessionButton,
    stopSpeakingButton,
    clearChatButton,
  });

  // 이벤트 리스너 등록
  if (sendButton) {
    sendButton.addEventListener("click", sendMessage);
  }

  if (startSessionButton) {
    startSessionButton.addEventListener("click", initializeAvatar);
  }

  if (stopSessionButton) {
    stopSessionButton.addEventListener("click", stopSession);
  }

  if (stopSpeakingButton) {
    stopSpeakingButton.addEventListener("click", stopSpeaking);
  }

  if (clearChatButton) {
    clearChatButton.addEventListener("click", clearChat);
  }

  // Enter 키로 메시지 전송
  document.getElementById("userInput").addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
});

// 아바타 초기화
async function initializeAvatar() {
  try {
    console.log("아바타 초기화 시작");
    document.getElementById("loader").style.display = "block";
    document.getElementById("startSessionButton").disabled = true;

    // Speech SDK 설정
    const speechKey = document
      .querySelector('meta[name="speech-key"]')
      .getAttribute("content");
    const speechRegion = document
      .querySelector('meta[name="speech-region"]')
      .getAttribute("content");

    console.log("Speech 설정:", {
      speechKeyExists: speechKey ? true : false,
      speechRegion,
    });

    if (!speechKey || !speechRegion) {
      throw new Error("Speech API 키 또는 지역이 설정되지 않았습니다.");
    }

    console.log("토큰 기반 인증 시작");

    // 토큰 기반 인증
    const tokenUrl = `https://${speechRegion}.api.cognitive.microsoft.com/sts/v1.0/issueToken`;
    console.log("토큰 URL:", tokenUrl);
    console.log(
      "Speech Key:",
      speechKey ? "설정됨 (길이: " + speechKey.length + ")" : "설정되지 않음"
    );

    try {
      const tokenResponse = await fetch(tokenUrl, {
        method: "POST",
        headers: {
          "Ocp-Apim-Subscription-Key": speechKey,
          "Content-type": "application/x-www-form-urlencoded",
          "Content-Length": "0",
        },
      });

      console.log("토큰 응답 상태:", tokenResponse.status);

      if (!tokenResponse.ok) {
        const errorText = await tokenResponse.text();
        console.error("토큰 오류 응답:", errorText);
        throw new Error(`토큰 획득 실패: ${tokenResponse.status}`);
      }

      const accessToken = await tokenResponse.text();
      console.log("토큰 획득 성공");

      // 토큰으로 SpeechConfig 초기화
      speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(
        accessToken,
        speechRegion
      );
      speechConfig.speechSynthesisVoiceName = "ko-KR-InJoonNeural";
    } catch (error) {
      console.error("토큰 요청 중 예외 발생:", error);
      throw error;
    }

    // ICE 서버 정보 직접 요청
    const iceServerUrl = `https://${speechRegion}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1`;
    const response = await fetch(iceServerUrl, {
      headers: {
        "Ocp-Apim-Subscription-Key": speechKey,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `ICE 서버 정보를 가져오지 못했습니다: 상태 코드 ${response.status}, 오류: ${errorText}`
      );
    }

    const iceServerInfo = await response.json();

    let iceServersConfig = [];
    // 다양한 응답 구조 처리
    if (iceServerInfo.iceServers && Array.isArray(iceServerInfo.iceServers)) {
      iceServersConfig = iceServerInfo.iceServers;
    } else if (iceServerInfo.Urls) {
      iceServersConfig = [
        {
          urls: Array.isArray(iceServerInfo.Urls)
            ? iceServerInfo.Urls
            : [iceServerInfo.Urls],
          username: iceServerInfo.Username || "",
          credential: iceServerInfo.Password || "",
        },
      ];
    } else if (iceServerInfo.urls) {
      iceServersConfig = [
        {
          urls: Array.isArray(iceServerInfo.urls)
            ? iceServerInfo.urls
            : [iceServerInfo.urls],
          username: iceServerInfo.username || "",
          credential: iceServerInfo.credential || "",
        },
      ];
    } else {
      console.warn("인식할 수 없는 ICE 서버 정보 구조:", iceServerInfo);
      iceServersConfig = [
        {
          urls: ["turn:relay.communication.microsoft.com:3478"],
          username: "",
          credential: "",
        },
      ];
    }

    console.log("최종 ICE 서버 설정:", iceServersConfig);

    peerConnection = new RTCPeerConnection({
      iceServers: iceServersConfig,
    });

    // 비디오/오디오 트랙 처리
    peerConnection.ontrack = function (event) {
      if (event.track.kind === "video") {
        const videoElement = document.createElement("video");
        videoElement.id = "avatarVideo";
        videoElement.srcObject = event.streams[0];
        videoElement.autoplay = true;
        videoElement.playsInline = true;

        const remoteVideoDiv = document.getElementById("remoteVideo");
        remoteVideoDiv.innerHTML = "";
        remoteVideoDiv.appendChild(videoElement);

        videoElement.onplaying = () => {
          console.log("아바타 비디오 연결됨");
          sessionActive = true;
          document.getElementById("stopSessionButton").disabled = false;
          document.getElementById("startSessionButton").disabled = true;
        };
      } else if (event.track.kind === "audio") {
        const audioElement = document.createElement("audio");
        audioElement.id = "avatarAudio";
        audioElement.srcObject = event.streams[0];
        audioElement.autoplay = true;
        document.body.appendChild(audioElement);
      }
    };

    // 트랜시버 추가
    peerConnection.addTransceiver("video", { direction: "sendrecv" });
    peerConnection.addTransceiver("audio", { direction: "sendrecv" });

    try {
      // 아바타 설정
      console.log("아바타 설정 생성 시작");

      // 유효한 아바타 설정 값
      const avatarCharacter = "Meg";
      const avatarStyle = "business";
      const avatarBgColor = "#FFFFFFFF";

      console.log("아바타 설정 값:", {
        avatarCharacter,
        avatarStyle,
        avatarBgColor,
      });

      // AvatarConfig 객체 생성
      const avatarConfig = new SpeechSDK.AvatarConfig(
        avatarCharacter,
        avatarStyle,
        avatarBgColor
      );

      console.log("아바타 설정 생성 완료");

      // 아바타 신시사이저 생성
      avatarSynthesizer = new SpeechSDK.AvatarSynthesizer(
        speechConfig,
        avatarConfig
      );
      console.log("AvatarSynthesizer 생성 완료");

      // 아바타 시작 및 WebRTC 연결 설정
      const result = await avatarSynthesizer.startAvatarAsync(peerConnection);

      if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
        console.log("아바타 세션이 성공적으로 시작되었습니다.");
        sessionActive = true; // 세션 상태 직접 설정
        document.getElementById("chatHistory").classList.remove("hidden");

        // 환영 메시지
        setTimeout(() => {
          if (avatarSynthesizer && sessionActive) {
            speak("안녕하세요! 무엇을 도와드릴까요?");
          } else {
            console.error(
              "아바타가 초기화되지 않았습니다. 세션 상태:",
              sessionActive
            );
            // 대체 메시지 표시
            addMessage(
              "system",
              "아바타 초기화에 문제가 있습니다. 텍스트로 계속 진행합니다."
            );
          }
        }, 1000);
      } else {
        throw new Error("아바타를 시작할 수 없습니다.");
      }
    } catch (error) {
      console.error("아바타 설정 생성 중 오류:", error);
      throw new Error("아바타 설정을 생성할 수 없습니다: " + error.message);
    }
  } catch (error) {
    console.error("아바타 초기화 중 오류:", error);
    addMessage("system", `오류: ${error.message}`);
  } finally {
    document.getElementById("loader").style.display = "none";
  }
}

// 메시지 전송
async function sendMessage() {
  const userInput = document.getElementById("userInput");
  const query = userInput.value.trim();
  if (!query) return;

  // UI 업데이트
  addMessage("user", query);
  userInput.value = "";
  document.getElementById("loader").style.display = "block";
  document.getElementById("sendButton").disabled = true;

  // WebSocket 연결 확인 및 재연결 시도
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.log("WebSocket 연결이 없거나 열려있지 않습니다. 재연결 시도...");
    connectWebSocket();
    // 연결 대기
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  // WebSocket으로 메시지 전송
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(query);
  } else {
    console.error("WebSocket 연결이 없습니다.");
    addMessage(
      "system",
      "서버 연결이 끊어졌습니다. 페이지를 새로고침 해주세요."
    );
    document.getElementById("loader").style.display = "none";
    document.getElementById("sendButton").disabled = false;
  }
}

// 아바타 말하기
async function speak(text) {
  if (!avatarSynthesizer || !sessionActive) {
    console.error("아바타가 초기화되지 않았습니다.");
    addMessage("assistant", text);
    return;
  }

  try {
    // 말하기 상태 업데이트
    isSpeaking = true;
    document.getElementById("stopSpeakingButton").disabled = false;

    // SSML 생성
    const ssml = `
         <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="ko-KR">
             <voice name="ko-KR-SunHiNeural">
                 ${text
                   .replace(/&/g, "&amp;")
                   .replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;")
                   .replace(/"/g, "&quot;")
                   .replace(/'/g, "&apos;")}
             </voice>
         </speak>`;

    // 아바타 말하기
    const result = await avatarSynthesizer.speakSsmlAsync(ssml);

    if (result.reason === SpeechSDK.ResultReason.SynthesizingAudioCompleted) {
      console.log("아바타 말하기 완료");
    } else {
      console.error("아바타 말하기 실패:", result);
    }
  } catch (error) {
    console.error("아바타 말하기 중 오류:", error);
    addMessage("system", `말하기 오류: ${error.message}`);
  } finally {
    // 자막 숨기기
    document.getElementById("subtitles").classList.add("hidden");

    // 말하기 상태 업데이트
    isSpeaking = false;
    document.getElementById("stopSpeakingButton").disabled = true;
  }
}

// 말하기 중지
function stopSpeaking() {
  if (avatarSynthesizer && isSpeaking) {
    avatarSynthesizer
      .stopSpeakingAsync()
      .then(() => {
        console.log("말하기 중지됨");
        isSpeaking = false;
        document.getElementById("stopSpeakingButton").disabled = true;
        document.getElementById("subtitles").classList.add("hidden");
      })
      .catch((error) => {
        console.error("말하기 중지 중 오류:", error);
      });
  }
}

// 세션 중지
function stopSession() {
  if (avatarSynthesizer) {
    // 말하기 중지
    if (isSpeaking) {
      stopSpeaking();
    }

    // 아바타 세션 종료
    avatarSynthesizer.close();
    avatarSynthesizer = null;
  }

  if (peerConnection) {
    // 피어 연결 종료
    peerConnection.close();
    peerConnection = null;
  }

  // 오디오 요소 제거
  const audioElement = document.getElementById("avatarAudio");
  if (audioElement) {
    audioElement.remove();
  }

  // 비디오 컨테이너 초기화
  document.getElementById("remoteVideo").innerHTML = "";

  // 상태 업데이트
  sessionActive = false;
  document.getElementById("stopSessionButton").disabled = true;
  document.getElementById("startSessionButton").disabled = false;
  document.getElementById("stopSpeakingButton").disabled = true;

  console.log("아바타 세션이 종료되었습니다.");
}

// 채팅 내역 지우기
function clearChat() {
  document.getElementById("chatHistory").innerHTML = "";
}

marked.setOptions({
  breaks: true, // 줄바꿈을 <br>로 변환
  gfm: true, // GitHub Flavored Markdown 활성화
  headerIds: false, // 헤더에 ID 자동 추가 비활성화
  // sanitize 옵션은 더 이상 사용되지 않습니다
  highlight: function (code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
});

// 메시지 추가
function addMessage(type, content) {
  const chatHistory = document.getElementById("chatHistory");
  const messageDiv = document.createElement("div");
  switch (type) {
    case "user":
      messageDiv.className = "user-message";
      messageDiv.textContent = `나: ${content}`;
      break;
    case "assistant":
      messageDiv.className = "assistant-message";
      messageDiv.innerHTML = marked.parse(`AI: ${content}`);
      break;
    case "assistant-text":
      messageDiv.className = "assistant-message";
      messageDiv.innerHTML = marked.parse(`Text AI: ${content}`);
      break;
    case "system":
      messageDiv.className = "error-message";
      messageDiv.textContent = content;
      break;
    default:
      messageDiv.textContent = content;
  }
  chatHistory.appendChild(messageDiv);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

// 오류 처리 함수
function handleError(error) {
  console.error("오류 발생:", error);
  addMessage("system", `오류가 발생했습니다: ${error.message}`);
  document.getElementById("loader").style.display = "none";
  document.getElementById("sendButton").disabled = false;
}

// WebSocket 재연결 함수
function reconnectWebSocket() {
  console.log("WebSocket 재연결 시도...");
  connectWebSocket();
}

// 아바타 상태 확인 함수
function checkAvatarStatus() {
  if (!sessionActive) {
    console.warn("아바타 세션이 활성화되지 않았습니다.");
    addMessage(
      "system",
      "아바타 세션이 활성화되지 않았습니다. 세션을 시작해주세요."
    );
    return false;
  }
  return true;
}

// 사용자 입력 유효성 검사 함수
function validateUserInput(input) {
  if (!input || input.trim().length === 0) {
    console.warn("유효하지 않은 사용자 입력");
    addMessage("system", "메시지를 입력해주세요.");
    return false;
  }
  return true;
}

// 아바타 설정 초기화 함수
function resetAvatarSettings() {
  sessionActive = false;
  isSpeaking = false;
  avatarSynthesizer = null;
  peerConnection = null;
  document.getElementById("stopSessionButton").disabled = true;
  document.getElementById("startSessionButton").disabled = false;
  document.getElementById("stopSpeakingButton").disabled = true;
  document.getElementById("remoteVideo").innerHTML = "";
  const audioElement = document.getElementById("avatarAudio");
  if (audioElement) {
    audioElement.remove();
  }
  console.log("아바타 설정이 초기화되었습니다.");
}

// 페이지 언로드 시 정리 작업
window.addEventListener("beforeunload", () => {
  if (sessionActive) {
    stopSession();
  }
  if (ws) {
    ws.close();
  }
});

// 디버그 모드 토글 함수
let debugMode = false;
function toggleDebugMode() {
  debugMode = !debugMode;
  console.log(`디버그 모드 ${debugMode ? "활성화" : "비활성화"}`);
  // 디버그 모드에 따른 추가 로직 구현
}

// 성능 모니터링 함수
function monitorPerformance() {
  const performanceData = {
    memory: performance.memory,
    timing: performance.timing,
    navigation: performance.navigation,
  };
  console.log("성능 데이터:", performanceData);
  // 성능 데이터 분석 및 최적화 로직 구현
}

// 주기적인 연결 상태 확인
setInterval(() => {
  if (ws && ws.readyState !== WebSocket.OPEN) {
    console.warn("WebSocket 연결이 끊어졌습니다. 재연결을 시도합니다.");
    reconnectWebSocket();
  }
}, 30000); // 30초마다 확인

// 사용자 활동 감지 및 세션 유지
let userActivityTimeout;
function resetUserActivityTimer() {
  clearTimeout(userActivityTimeout);
  userActivityTimeout = setTimeout(() => {
    console.log("사용자 비활성 감지. 세션을 종료합니다.");
    stopSession();
  }, 300000); // 5분 후 세션 종료
}
document.addEventListener("mousemove", resetUserActivityTimer);
document.addEventListener("keypress", resetUserActivityTimer);

// 브라우저 호환성 확인
function checkBrowserCompatibility() {
  const isWebRTCSupported =
    navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
  const isWebSocketSupported = "WebSocket" in window;
  if (!isWebRTCSupported || !isWebSocketSupported) {
    console.warn("브라우저가 필요한 기능을 지원하지 않습니다.");
    addMessage(
      "system",
      "현재 브라우저에서 일부 기능이 지원되지 않을 수 있습니다. 최신 브라우저를 사용해주세요."
    );
  }
}

// 초기화 함수
function initialize() {
  checkBrowserCompatibility();
  connectWebSocket();
  resetUserActivityTimer();
  if (debugMode) {
    monitorPerformance();
  }
}

// 페이지 로드 시 초기화 실행
window.addEventListener("load", initialize);
