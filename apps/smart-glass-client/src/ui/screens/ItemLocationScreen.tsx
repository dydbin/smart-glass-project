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

const dummyLocations = [
  {
    id: 1,
    location: '위치데이터가 표시될 것입니다',
    time: '시간데이터가 표시될 것입니다',
  },
];

export default function ItemLocationScreen() {
  const navigation = useNavigation<any>();

  return (
    <SafeAreaView style={commonStyles.screen}>
      <View style={commonStyles.header}>
        <Pressable onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>←</Text>
        </Pressable>

        <Text style={commonStyles.headerTitle}>위치 기록</Text>

        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.titleBox}>
          <Text style={styles.itemName}>선택한 물품</Text>
          <Text style={styles.description}>
            이 물품이 발견되었던 위치 기록입니다.
          </Text>
        </View>

        {dummyLocations.map((item) => (
          <View key={item.id} style={commonStyles.card}>
            <Text style={styles.location}>{item.location}</Text>
            <Text style={styles.time}>{item.time}</Text>
          </View>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  content: {
    padding: 16,
    gap: 12,
  },

  titleBox: {
    marginBottom: 8,
  },

  itemName: {
    fontSize: 22,
    fontWeight: '800',
    color: colors.text,
    marginBottom: 6,
  },

  description: {
    fontSize: 14,
    color: colors.subText,
  },

  location: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.text,
    marginBottom: 6,
  },

  time: {
    fontSize: 13,
    color: colors.subText,
  },

  backButton: {
    fontSize: 20,
    color: colors.text,
    width: 24,
  },
});