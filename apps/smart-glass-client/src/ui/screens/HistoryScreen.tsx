import React from 'react';
import { useNavigation } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  ScrollView,
} from 'react-native';

import { commonStyles } from '../styles/commonStyles';
import { colors } from '../styles/colors';
import { useItemContext } from '../context/ItemContext';

const formatTime = (timestamp: number) => {
  const now = Date.now();
  const diff = Math.floor((now - timestamp) / 1000);

  if (diff < 60) return '방금 전';
  if (diff < 3600) return `${Math.floor(diff / 60)}분 전`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}시간 전`;

  return `${Math.floor(diff / 86400)}일 전`;
};


export default function HistoryScreen() {
  const navigation = useNavigation<any>();
  const { itemCounts } = useItemContext();

  const sortedItems = Object.entries(itemCounts).sort(
    (a, b) => b[1].lastTime - a[1].lastTime
  );

  return (
    <SafeAreaView style={commonStyles.screen}>
      <View style={commonStyles.header}>
        <Pressable onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>←</Text>
        </Pressable>

        <Text style={commonStyles.headerTitle}>히스토리</Text>

        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={styles.list}>
        <Text style={styles.sectionTitle}>자주 찾은 물건</Text>

        {sortedItems.length === 0 ? (
          <View style={commonStyles.card}>
            <Text style={styles.emptyTitle}>아직 히스토리가 없어요</Text>
            <Text style={styles.emptyText}>
              채팅에서 지갑, 이어폰, 열쇠 같은 물건을 질문하면 여기에 기록됩니다.
            </Text>
          </View>
        ) : (
          sortedItems.map(([item, data]) => (
            <Pressable
              key={item}
              style={commonStyles.card}
              onPress={() =>
                navigation.navigate('ItemLocation', { itemName: item })
              }
            >
              <View style={styles.cardTop}>
                <Text style={styles.cardTitle}>{item}</Text>
                <Text style={styles.time}>{formatTime(data.lastTime)}</Text>
              </View>

              <Text style={styles.preview}>
                가장 최근에 어디서 찾았는지 장소 정보 출력
              </Text>
            </Pressable>
          ))
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  list: {
    padding: 16,
    gap: 12,
  },

  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.text,
    marginBottom: 4,
  },

  cardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },

  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
  },

  time: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.subText,
  },

  preview: {
    fontSize: 14,
    color: colors.subText,
  },

  emptyTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.text,
    marginBottom: 6,
  },

  emptyText: {
    fontSize: 14,
    color: colors.subText,
    lineHeight: 20,
  },

  backButton: {
    fontSize: 20,
    color: colors.text,
    width: 24,
  },
});