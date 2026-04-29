import { Injectable } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class MediaService {
  async uploadImage(file: Express.Multer.File, body: any) {
    try {
      if (!file) {
        return {
          upload_status: 'fail',
          error: 'file is missing',
        };
      } 
      // const {timestamp, device_id, ...sceneMetadata} = body;
      const {timestamp, device_id, optional_scene_metadata} = body;

      const uploadDir = 'uploads';

      if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir);
      }

      const fileName = Date.now() + '-' + file.originalname;
      const filePath = path.join(uploadDir, fileName);

      fs.writeFileSync(filePath, file.buffer); // 나중에 S3에 연결되게끔 바꾸기

      const image_id = 1; //나중에 DB연결하면 고유 ID로 저장되게끔 바꾸기

      return {
        upload_status: 'success',
        image_id,
        object_candidates: [], // 나중에 AI 답변? 들어갈 자리
      };
      
    } catch (err) {
      if (err instanceof Error) {
        return {
          upload_status: 'fail',
          error: err.message,
        };
      }
    }
  }
}