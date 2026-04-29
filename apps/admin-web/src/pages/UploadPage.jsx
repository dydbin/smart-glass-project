import { useState } from 'react';

export default function UploadAdminPage() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('대기 중');
  const [preview, setPreview] = useState(null);
  const [logs, setLogs] = useState([]);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);

    if (selected) {
      const url = URL.createObjectURL(selected);
      setPreview(url);
    }
  };

  const uploadImage = async () => {
    if (!file) {
      setStatus('파일을 선택하세요 ❗');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);
    formData.append('timestamp', new Date().toISOString());
    formData.append('device_id', 'admin_test');

    setStatus('업로드 중... ⏳');

    try {
      const res = await fetch('http://localhost:3000/upload-image', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (data.upload_status === 'success') {
        setStatus('업로드 성공 ✅');
        setLogs((prev) => [
          { time: new Date().toLocaleTimeString(), result: '성공' },
          ...prev,
        ]);
      } else {
        setStatus('업로드 실패 ❌');
        setLogs((prev) => [
          { time: new Date().toLocaleTimeString(), result: '실패' },
          ...prev,
        ]);
      }
    } catch (err) {
      setStatus('서버 오류 ❌');
    }
  };

  return (
    <div style={{ padding: '30px', fontFamily: 'sans-serif' }}>
      <h1>📊 Smart Glass Admin</h1>

      {/* 업로드 영역 */}
      <div style={{ marginBottom: '20px' }}>
        <h3>이미지 업로드</h3>
        <input type="file" onChange={handleFileChange} />
        <button onClick={uploadImage} style={{ marginLeft: '10px' }}>
          업로드
        </button>
      </div>

      {/* 미리보기 */}
      {preview && (
        <div style={{ marginBottom: '20px' }}>
          <h3>미리보기</h3>
          <img src={preview} alt="preview" width="200" />
        </div>
      )}

      {/* 상태 */}
      <div style={{ marginBottom: '20px' }}>
        <h3>업로드 상태</h3>
        <p>{status}</p>
      </div>

      {/* 로그 */}
      <div>
        <h3>최근 업로드 로그</h3>
        <ul>
          {logs.map((log, idx) => (
            <li key={idx}>
              [{log.time}] {log.result}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}