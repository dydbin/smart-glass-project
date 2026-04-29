import React, { useState } from 'react';
import { useNavigation } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  View,
  Text,
  StyleSheet,
  Switch,
  Pressable,
} from 'react-native';

import { commonStyles } from '../styles/commonStyles';

export default function SettingsScreen() {
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [notificationEnabled, setNotificationEnabled] = useState(true);
  const navigation = useNavigation<any>();

  return (
    <SafeAreaView style={commonStyles.screen}>
      <View style={commonStyles.header}>
        <Pressable onPress={() => navigation.goBack()}>
            <Text style={styles.backButton}>←</Text>
        </Pressable>

        <Text style={commonStyles.headerTitle}>설정</Text>

        <View style={{ width: 24 }} />
        </View>

      <View style={styles.content}>
        <View style={styles.item}>
          <Text style={styles.label}>음성 입력 사용</Text>
          <Switch
            value={voiceEnabled}
            onValueChange={setVoiceEnabled}
          />
        </View>

        <View style={styles.item}>
          <Text style={styles.label}>알림 허용</Text>
          <Switch
            value={notificationEnabled}
            onValueChange={setNotificationEnabled}
          />
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  content: {
    padding: 16,
    gap: 12,
  },

  item: {
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  label: {
    fontSize: 16,
  },
  backButton: {
    fontSize: 20,
    color: '#111827',
    width: 24,
  },
});