import React, { useMemo, useRef, useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  TextInput,
  ScrollView,
} from 'react-native';
import Sidebar from '../components/SideBar';

import { Image } from 'react-native';
import logo from '../icon/logo.png';
import micIcon from '../icon/mic.png';

import { commonStyles } from '../styles/commonStyles';

import { useItemContext } from '../context/ItemContext';

type Message = {
  id: number;
  sender: 'bot' | 'user';
  text: string;
};

const getDummyReply = (input: string) => {
  return `"${input}"에 대한 더미 응답입니다. 나중에 여기에 실제 AI 응답이 연결될 예정입니다.`;
  //TODO: AI 응답과 연동하여 실제 답변이 나오도록 수정 필요
};

export default function ChatScreen() {
  const { addItem } = useItemContext();
  const knownItems = ['지갑', '이어폰', '열쇠', '가방', '안경', '충전기', '텀블러'];
  //히스토리 확인을 위한 더미 데이터
  //TODO: 실제 AI 응답과 연동하여 사용자가 언급한 아이템을 기록하도록 수정 필요
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [searchText, setSearchText] = useState('');

  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isBotTyping, setIsBotTyping] = useState(false);

  const scrollViewRef = useRef<ScrollView>(null);
  const messagePositions = useRef<Record<number, number>>({});
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);

  const matchedMessageIds = useMemo(() => {
    const keyword = searchText.trim().toLowerCase();
    if (!keyword) return [];

    return messages
      .filter((message) => message.text.toLowerCase().includes(keyword))
      .map((message) => message.id);
  }, [searchText, messages]);

  const goToMatch = (index: number) => {
    if (matchedMessageIds.length === 0) return;

    const safeIndex =
      ((index % matchedMessageIds.length) + matchedMessageIds.length) %
      matchedMessageIds.length;

    const targetId = matchedMessageIds[safeIndex];
    const y = messagePositions.current[targetId] ?? 0;

    scrollViewRef.current?.scrollTo({
      y: Math.max(y - 20, 0),
      animated: true,
    });

    setCurrentMatchIndex(safeIndex);
  };

  React.useEffect(() => {
    if (matchedMessageIds.length > 0) {
      setCurrentMatchIndex(0);
      setTimeout(() => goToMatch(0), 50);
    } else {
      setCurrentMatchIndex(0);
    }
  }, [searchText, messages]);

  const scrollToBottom = () => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  };

  const handleSend = () => {
    
    const trimmed = inputText.trim();
    if (!trimmed) return;

    const foundItem = knownItems.find((item) => trimmed.includes(item));

    if (foundItem) {
      addItem(foundItem); 
    }

    const userMessage: Message = {
      id: Date.now(),
      sender: 'user',
      text: trimmed,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsBotTyping(true);
    scrollToBottom();

    setTimeout(() => {
      const botMessage: Message = {
        id: Date.now() + 1,
        sender: 'bot',
        text: getDummyReply(trimmed),
      };

      setMessages((prev) => [...prev, botMessage]);
      setIsBotTyping(false);
      scrollToBottom();
    }, 700);
  };

  const renderHighlightedText = (
    text: string,
    sender: 'bot' | 'user'
  ) => {
    const keyword = searchText.trim();

    const baseTextStyle =
      sender === 'user' ? styles.userMessageText : styles.botMessageText;

    if (!keyword) {
      return <Text style={baseTextStyle}>{text}</Text>;
    }

    const lowerText = text.toLowerCase();
    const lowerKeyword = keyword.toLowerCase();

    if (!lowerText.includes(lowerKeyword)) {
      return <Text style={baseTextStyle}>{text}</Text>;
    }

    const parts: React.ReactNode[] = [];
    let startIndex = 0;
    let matchIndex = 0;

    while (startIndex < text.length) {
      const foundIndex = lowerText.indexOf(lowerKeyword, startIndex);

      if (foundIndex === -1) {
        parts.push(
          <Text key={`text-${startIndex}`}>{text.slice(startIndex)}</Text>
        );
        break;
      }

      if (foundIndex > startIndex) {
        parts.push(
          <Text key={`text-${startIndex}`}>
            {text.slice(startIndex, foundIndex)}
          </Text>
        );
      }

      parts.push(
        <Text key={`match-${matchIndex}`} style={styles.highlightText}>
          {text.slice(foundIndex, foundIndex + keyword.length)}
        </Text>
      );

      startIndex = foundIndex + keyword.length;
      matchIndex += 1;
    }

    return <Text style={baseTextStyle}>{parts}</Text>;
  };

  return (
    <SafeAreaView style={commonStyles.screen}>
      <View style={commonStyles.header}>
        {isSearchMode ? (
          <>
            <TextInput
              value={searchText}
              onChangeText={setSearchText}
              placeholder="채팅 내용 검색"
              placeholderTextColor="#9CA3AF"
              style={styles.searchInput}
              autoFocus
            />
            <Pressable
              onPress={() => {
                setIsSearchMode(false);
                setSearchText('');
              }}
            >
              <Text style={styles.cancel}>취소</Text>
            </Pressable>
          </>
        ) : (
          <>
            <Pressable
              style={styles.menuButton}
              onPress={() => setSidebarVisible(true)}
            >
              <Text style={styles.menuText}>☰</Text>
            </Pressable>

            <View style={styles.headerCenter}>
              <Image source={logo} style={styles.headerLogo} />
            </View>

            <Pressable
              style={styles.searchButton}
              onPress={() => setIsSearchMode(true)}
            >
              <Text style={styles.searchIcon}>⌕</Text>
            </Pressable>
          </>
        )}
      </View>

      {isSearchMode && searchText.trim() ? (
        <View style={styles.searchInfoBar}>
          <Text style={styles.searchInfoText}>
            {matchedMessageIds.length > 0
              ? `${currentMatchIndex + 1} / ${matchedMessageIds.length}`
              : '검색 결과 0건'}
          </Text>

          <View style={styles.searchActions}>
            <Pressable
              style={styles.searchMoveButton}
              onPress={() => goToMatch(currentMatchIndex - 1)}
            >
              <Text style={styles.searchMoveText}>이전</Text>
            </Pressable>

            <Pressable
              style={styles.searchMoveButton}
              onPress={() => goToMatch(currentMatchIndex + 1)}
            >
              <Text style={styles.searchMoveText}>다음</Text>
            </Pressable>
          </View>
        </View>
      ) : null}

      <ScrollView
        ref={scrollViewRef}
        contentContainerStyle={styles.chatArea}
      >
        {messages.length === 0 ? (
          <View style={styles.emptyBox}>
            <Text style={styles.emptyTitle}>환영합니다</Text>
            <Text style={styles.emptyText}>
              찾고자 하는 물건이 있으시다면 말씀해주세요
            </Text>
          </View>
        ) : null}

        {messages.map((message) => {
          const isMatched = matchedMessageIds.includes(message.id);

          return (
            <View
              key={message.id}
              onLayout={(event) => {
                messagePositions.current[message.id] =
                  event.nativeEvent.layout.y;
              }}
              style={[
                message.sender === 'bot'
                  ? styles.botMessage
                  : styles.userMessage,
                isMatched && styles.matchedMessage,
                isMatched &&
                  matchedMessageIds[currentMatchIndex] === message.id &&
                  styles.activeMatchedMessage,
              ]}
            >
              {renderHighlightedText(message.text, message.sender)}
            </View>
          );
        })}

        {isBotTyping ? (
          <View style={styles.botMessage}>
            <Text style={styles.botMessageText}>입력 중...</Text>
          </View>
        ) : null}
      </ScrollView>

      {/* <View style={styles.frequentQuestionsSection}>
        <Text style={styles.frequentQuestionsTitle}>최근 자주 물어본 질문</Text>
        <View style={styles.quickQuestions}>
          {frequentQuestions.map((question) => (
            <Pressable
              key={question}
              style={styles.quickButton}
              onPress={() => setInputText(question)}
            >
              <Text style={styles.quickButtonText}>{question}</Text>
            </Pressable>
          ))}
        </View>
      </View> */}

      <View style={styles.inputArea}>
        <TextInput
          value={inputText}
          onChangeText={setInputText}
          placeholder="메시지를 입력하세요"
          placeholderTextColor="#9CA3AF"
          style={styles.input}
          onSubmitEditing={handleSend}
        />

        <Pressable style={styles.sendButton} onPress={handleSend}>
          {inputText.trim() ? (
            <Text style={styles.actionButtonText}>전송</Text>
          ) : (
            <Image source={micIcon} style={styles.micIcon} />
          )}
        </Pressable>
      </View>

      <Sidebar
        visible={sidebarVisible}
        onClose={() => setSidebarVisible(false)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  menuButton: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },

  menuText: {
    fontSize: 20,
    color: '#111827',
  },

  searchButton: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },

  searchIcon: {
    fontSize: 20,
    color: '#111827',
  },

  searchInput: {
    flex: 1,
    height: 40,
    backgroundColor: '#F3F4F6',
    borderRadius: 20,
    paddingHorizontal: 14,
    fontSize: 15,
    color: '#111827',
  },

  cancel: {
    marginLeft: 10,
    color: '#2563EB',
    fontWeight: '500',
    fontSize: 16,
  },

  searchInfoBar: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#FFF7D6',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  searchInfoText: {
    fontSize: 13,
    color: '#7C5E10',
    fontWeight: '600',
  },

  searchActions: {
    flexDirection: 'row',
    gap: 8,
  },

  searchMoveButton: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },

  searchMoveText: {
    fontSize: 12,
    color: '#374151',
    fontWeight: '600',
  },

  chatArea: {
    padding: 16,
    gap: 12,
  },

  emptyBox: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 18,
    alignItems: 'center',
    marginTop: 8,
  },

  emptyTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#111827',
    marginBottom: 6,
  },

  emptyText: {
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
    lineHeight: 20,
  },

  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 16,
    maxWidth: '80%',
  },

  botMessageText: {
    fontSize: 15,
    color: '#111827',
    lineHeight: 22,
  },

  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#2563EB',
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 16,
    maxWidth: '80%',
  },

  userMessageText: {
    fontSize: 15,
    color: '#FFFFFF',
    lineHeight: 22,
  },

  matchedMessage: {
    borderWidth: 2,
    borderColor: '#FACC15',
  },

  activeMatchedMessage: {
    borderColor: '#EAB308',
  },

  highlightText: {
    backgroundColor: '#FDE68A',
    color: '#111827',
    fontWeight: '700',
  },

  frequentQuestionsSection: {
    paddingHorizontal: 16,
    paddingBottom: 12,
  },

  frequentQuestionsTitle: {
    fontSize: 13,
    color: '#6B7280',
    marginBottom: 8,
    fontWeight: '600',
  },

  quickQuestions: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },

  quickButton: {
    backgroundColor: '#EEF2FF',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 999,
  },

  quickButtonText: {
    fontSize: 13,
    color: '#374151',
  },

  inputArea: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 20,
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
    backgroundColor: '#FFFFFF',
    gap: 10,
  },

  input: {
    flex: 1,
    height: 48,
    backgroundColor: '#F3F4F6',
    borderRadius: 24,
    paddingHorizontal: 16,
    fontSize: 15,
    color: '#111827',
  },

  sendButton: {
    minWidth: 64,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#f1f1f1',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 16,
  },

  sendButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
  },

  headerCenter: {
    flex: 1,
    alignItems: 'center',
  },

  headerLogo: {
    width: 40,
    height: 40,
    resizeMode: 'contain',
  },

  actionButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
  },

  micIcon: {
    width: 22,
    height: 22,
    resizeMode: 'contain',
  },
});