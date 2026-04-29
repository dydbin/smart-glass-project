import React from 'react';
import { useNavigation } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { View, Text, StyleSheet, Pressable } from 'react-native';

import { commonStyles } from '../styles/commonStyles';
import { colors } from '../styles/colors';

export default function ProfileScreen() {
  const navigation = useNavigation<any>();

  return (
    <SafeAreaView style={commonStyles.screen}>
      <View style={commonStyles.header}>
        <Pressable onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>←</Text>
        </Pressable>

        <Text style={commonStyles.headerTitle}>사용자 정보</Text>

        <View style={{ width: 24 }} />
      </View>

      <View style={styles.content}>
        <View style={styles.avatar}>
          <Text style={styles.avatarIcon}>👤</Text>
        </View>
        {/*TODO: 사용자 프로필 사진 연동 필요 */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>이메일</Text>
          <View style={styles.inputBox}>
            <Text style={styles.value}>smart1336@gmail.com</Text>
          </View>
        </View>
        {/*TODO : 사용자 이메일 연동 필요 */}
        <View style={styles.fieldGroup}>
          <Text style={styles.label}>연결된 스마트 글라스</Text>
          <View style={styles.inputBox}>
            <Text style={styles.value}>glass-1336</Text>
          </View>
        </View>
        {/*TODO: 연결된 스마트 글라스 정보 연동 필요 */}

        <Text style={styles.syncText}>최근 데이터 동기화: 2분 전</Text> 
      </View>
    </SafeAreaView>
    //TODO: 실제 사용자 데이터 연동 필요
  );
}

const styles = StyleSheet.create({
  content: {
    flex: 1,
    paddingHorizontal: 36,
    paddingTop: 48,
  },

  avatar: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: '#D9D9D9',
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 36,
  },

  avatarIcon: {
    fontSize: 42,
  },

  fieldGroup: {
    marginBottom: 16,
  },

  label: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.text,
    marginBottom: 8,
  },

  inputBox: {
    height: 46,
    borderRadius: 8,
    backgroundColor: '#D1D1D1',
    justifyContent: 'center',
    paddingHorizontal: 14,
  },

  value: {
    fontSize: 15,
    color: colors.text,
    textDecorationLine: 'underline',
  },

  syncText: {
    fontSize: 12,
    color: colors.subText,
    textAlign: 'right',
    marginTop: -6,
  },

  backButton: {
    fontSize: 20,
    color: colors.text,
    width: 24,
  },
});