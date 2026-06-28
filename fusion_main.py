import carla, math, queue, random, time, cv2, numpy as np, matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial.distance import mahalanobis, cosine 
from sklearn.cluster import DBSCAN 

class MetricsTracker:
    def __init__(self):
        self.TP = self.FP = self.FN = self.total_frames = self.id_switches = 0
        self.ATE_sum = self.IoU_sum = 0.0
        self.gt_to_track_map = {} 

    def _calculate_bev_iou(self, t_pos, gt):
        x_left, y_top = max(gt['x'] - gt['length']/2, t_pos[0] - 2.0), max(gt['y'] - gt['width']/2, t_pos[1] - 1.0)
        x_right, y_bottom = min(gt['x'] + gt['length']/2, t_pos[0] + 2.0), min(gt['y'] + gt['width']/2, t_pos[1] + 1.0)
        if x_right < x_left or y_bottom < y_top: return 0.0
        inter = (x_right - x_left) * (y_bottom - y_top)
        return inter / float((gt['length'] * gt['width']) + 8.0 - inter)

    def evaluate_frame(self, predicted_tracks, gt_objects, dist_thresh=4.0):
        self.total_frames += 1
        matched_gt = set()
        for t in [t for t in predicted_tracks if t.missed_frames <= 3]:
            best_gt, min_dist = None, dist_thresh
            for gt in [g for g in gt_objects if g['id'] not in matched_gt]:
                d = math.hypot(t.position[0] - gt['x'], t.position[1] - gt['y'])
                if d < min_dist: best_gt, min_dist = gt, d
            if best_gt:
                self.TP += 1; matched_gt.add(best_gt['id'])
                self.ATE_sum += min_dist
                self.IoU_sum += self._calculate_bev_iou(t.position, best_gt)
                if best_gt['id'] in self.gt_to_track_map and self.gt_to_track_map[best_gt['id']] != t.id:
                    self.id_switches += 1
                self.gt_to_track_map[best_gt['id']] = t.id
            else: self.FP += 1  
        self.FN += len(gt_objects) - len(matched_gt)
        
    def print_summary(self):
        prec = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0
        rec = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        ate = self.ATE_sum / self.TP if self.TP > 0 else 0.0
        iou = self.IoU_sum / self.TP if self.TP > 0 else 0.0
        print(f"\n{'='*55}\n 🏁 SIMULATION EVALUATION METRICS 🏁 \n{'='*55}")
        print(f"Frames: {self.total_frames} | TP: {self.TP} | FP: {self.FP} | FN: {self.FN}")
        print(f"Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")
        print(f"Avg ATE: {ate:.2f}m | Avg BEV IoU: {iou*100:.2f}% | ID Switches: {self.id_switches}\n{'='*55}\n")

    def plot_metrics(self):
        prec = (self.TP / (self.TP + self.FP) * 100) if (self.TP + self.FP) > 0 else 0.0
        rec = (self.TP / (self.TP + self.FN) * 100) if (self.TP + self.FN) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        iou = (self.IoU_sum / self.TP * 100) if self.TP > 0 else 0.0
        ate = self.ATE_sum / self.TP if self.TP > 0 else 0.0

        fig, axs = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Sensor Fusion Tracking Performance', fontsize=16, fontweight='bold')
        
        b1 = axs[0].bar(['Precision', 'Recall', 'F1-Score', 'Avg IoU'], [prec, rec, f1, iou], color=['#4CAF50', '#2196F3', '#9C27B0', '#00BCD4'])
        axs[0].set(ylim=(0, 105), ylabel='Percentage (%)', title='Accuracy')
        for b in b1: axs[0].text(b.get_x() + b.get_width()/2., b.get_height()+2, f'{b.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')

        b2 = axs[1].bar(['Avg Translation Error'], [ate], color='#FF9800')
        axs[1].set(ylim=(0, max(2.0, ate + 1.0)), ylabel='Meters (m)', title='Localization Error')
        for b in b2: axs[1].text(b.get_x() + b.get_width()/2., b.get_height()+0.1, f'{b.get_height():.2f} m', ha='center', va='bottom', fontweight='bold')

        b3 = axs[2].bar(['Total ID Switches'], [self.id_switches], color='#F44336')
        axs[2].set(ylim=(0, max(5, self.id_switches + 5)), ylabel='Count', title='Tracking Stability')
        for b in b3: axs[2].text(b.get_x() + b.get_width()/2., b.get_height()+0.5, f'{int(b.get_height())}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout(); plt.savefig('performance_metrics.png', dpi=300); plt.show()

class ExtendedKalmanFilter:
    def __init__(self, dt, init_x, init_y, init_v, init_yaw):
        self.dt = dt
        self.x = np.array([init_x, init_y, init_v, init_yaw, 0.0]).reshape(5, 1)
        self.P, self.Q = np.eye(5) * 5.0, np.eye(5) * 0.05 
        self.P[4,4] = 10.0 
        self.R_pos, self.R_vel = np.eye(2) * 1.0, np.eye(3) * 1.0 

    def predict(self):
        px, py, v, yaw, yaw_rate = self.x.flatten()
        if abs(yaw_rate) > 0.001:
            nx = px + (v/yaw_rate) * (math.sin(yaw + yaw_rate*self.dt) - math.sin(yaw))
            ny = py + (v/yaw_rate) * (math.cos(yaw) - math.cos(yaw + yaw_rate*self.dt))
        else:
            nx, ny = px + v * math.cos(yaw) * self.dt, py + v * math.sin(yaw) * self.dt
            
        self.x[:2], self.x[3] = [[nx], [ny]], yaw + yaw_rate * self.dt
        F = np.eye(5)
        F[0, 2], F[1, 2] = math.cos(yaw) * self.dt, math.sin(yaw) * self.dt
        F[0, 3], F[1, 3] = -v * math.sin(yaw) * self.dt, v * math.cos(yaw) * self.dt
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def _update_core(self, z, H, R):
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(5) - (K @ H)) @ self.P
        return self.x

    def update(self, mx, my):
        H = np.zeros((2, 5)); H[0, 0] = H[1, 1] = 1
        return self._update_core(np.array([mx, my]).reshape(2, 1), H, self.R_pos)
    
    def update_with_velocity(self, mx, my, mv):
        H = np.zeros((3, 5)); H[0, 0] = H[1, 1] = H[2, 2] = 1
        return self._update_core(np.array([mx, my, mv]).reshape(3, 1), H, self.R_vel)

@dataclass
class TrackedObject:
    id: int; class_label: str; confidence: float; bbox: tuple; position: np.ndarray 
    velocity_vec: np.ndarray; velocity: float; yaw: float; kf: ExtendedKalmanFilter 
    features: np.ndarray = None; missed_frames: int = 0; hit_streak: int = 0
    confirmed: bool = False; sensor_source: str = "Fused"; lidar_pts: int = 0

class AdvancedTracker:
    def __init__(self, max_frames_lost=3):
        self.tracks = {}; self.next_id = 1; self.max_frames_lost = max_frames_lost
        self.dt = 1.0 / 30.0; self.gating_threshold = 5.99

    def predict_only(self):
        for t in self.tracks.values(): t.kf.predict(); t.position = t.kf.x[:2].flatten()
        return [t for t in self.tracks.values() if t.confirmed]

    def get_mahalanobis_dist(self, track, meas_pos):
        H = np.zeros((2, 5)); H[0, 0] = H[1, 1] = 1
        S = H @ track.kf.P @ H.T + track.kf.R_pos
        try: return mahalanobis(np.array(meas_pos), (H @ track.kf.x).flatten(), np.linalg.inv(S))
        except: return 100.0

    def update(self, measurements, ego_speed):
        used_tracks, updated_objects = set(), []
        for t in self.tracks.values(): t.kf.predict(); t.position = t.kf.x[:2].flatten()
        
        for m in measurements:
            mx, my = m['pos']
            best_id, best_cost = -1, 1000.0 
            
            for tid, t in self.tracks.items():
                if tid in used_tracks: continue
                m_dist = self.get_mahalanobis_dist(t, [mx, my])
                if m_dist > self.gating_threshold: continue
                reid_dist = cosine(t.features, m['features']) if t.features is not None and m.get('features') is not None else 0.0
                if math.isnan(reid_dist): reid_dist = 0.0
                cost = m_dist + (reid_dist * 5.0) 
                if cost < best_cost: best_cost, best_id = cost, tid
            
            if best_id != -1:
                t = self.tracks[best_id]
                t.missed_frames = 0; t.hit_streak += 1 
                if m.get('rad_vel') is not None:
                    mv = abs(m['rad_vel'])
                    state = t.kf.update(mx, my) if t.confirmed and abs(mv - t.velocity) > 10.0 else t.kf.update_with_velocity(mx, my, mv)
                else: state = t.kf.update(mx, my)
                    
                t.position = state[:2].flatten()
                v, yaw = float(state[2].item()), float(state[3].item())
                t.velocity_vec, t.velocity, t.yaw = np.array([v * math.cos(yaw), v * math.sin(yaw)]), abs(v), math.degrees(yaw)
                t.bbox, t.confidence, t.lidar_pts, t.sensor_source = m['bbox'], m['conf'], m['lidar_pts'], m['source']
                if m.get('features') is not None: t.features = m['features']
                
                if t.hit_streak >= (2 if "Fus" in t.sensor_source else 3): t.confirmed = True
                
                used_tracks.add(best_id)
                if t.confirmed: updated_objects.append(t)
            else:
                init_vx = -m['rad_vel'] if m.get('rad_vel') is not None else -ego_speed
                new_obj = TrackedObject(
                    id=self.next_id, class_label=m['label'], confidence=m['conf'], bbox=m['bbox'], 
                    position=np.array([mx, my]), velocity_vec=np.array([init_vx, 0.0]), velocity=abs(init_vx), 
                    yaw=math.degrees(math.pi if init_vx < 0 else 0.0), kf=ExtendedKalmanFilter(self.dt, mx, my, abs(init_vx), math.pi if init_vx < 0 else 0.0), 
                    hit_streak=1, sensor_source=m['source'], lidar_pts=m['lidar_pts'], features=m.get('features')
                )
                self.tracks[self.next_id] = new_obj; self.next_id += 1
        
        cleanup = []
        for tid, t in self.tracks.items():
            if tid not in used_tracks:
                t.missed_frames += 1
                if t.confirmed and t.missed_frames <= self.max_frames_lost: updated_objects.append(t)
                else: cleanup.append(tid)
        for tid in cleanup: del self.tracks[tid]
        return updated_objects

class SensorFusionEngine:
    def __init__(self, width, height, fov):
        self.width, self.height, self.fov, self.frame_count = width, height, fov, 0
        from ultralytics import YOLO
        self.model = YOLO("yolov8m.pt") 
        focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]])
        self.tracker = AdvancedTracker()

    def process(self, cam_data, lid_data, rad_data, ego_speed):
        self.frame_count += 1
        l_clusters, l3d = self._lid_independent(lid_data)
        r_points = self._rad_independent(rad_data)
        arr = np.reshape(np.frombuffer(cam_data.raw_data, dtype=np.uint8), (cam_data.height, cam_data.width, 4))[:,:,:3].copy()
        dets = self._cam(arr)
        meas, used_lid, used_rad = [], set(), set()
        
        # --- NEW STEP 1 PRE-COMPUTATION: Project all forward-facing LiDAR points to 2D ---
        valid_lid = l3d[:, 0] > 1.0 
        l3d_valid = l3d[valid_lid]
        if len(l3d_valid) > 0:
            u_all = (self.K[0,0] * -l3d_valid[:, 1] / l3d_valid[:, 0]) + self.K[0,2]
            v_all = (self.K[0,0] * -l3d_valid[:, 2] / l3d_valid[:, 0]) + self.K[1,2]
        else:
            u_all, v_all = np.array([]), np.array([])
        
        for d in dets:
            x1, y1, x2, y2 = d['box']
            cx_px = (x1+x2)/2
            source, dist, rad_vel, matched_lid_pts = "Camera", 0.0, None, 0
            
            # --- NEW STEP 1 CORE: Point-in-Box Median Depth Extraction ---
            in_box = (u_all >= x1) & (u_all <= x2) & (v_all >= y1) & (v_all <= y2)
            pts_in_box = l3d_valid[in_box]
            
            if len(pts_in_box) >= 3: # Need at least 3 points for a reliable median
                dist = float(np.median(pts_in_box[:, 0]))
                matched_lid_pts = len(pts_in_box)
                source = "Fus"
            
            # --- PREVIOUS MARGIN LOGIC: Expanded margin to prevent redundant tracking ---
            mx, my = (x2 - x1) * 1.2, (y2 - y1) * 1.2
            
            for i, lc in enumerate(l_clusters):
                if i not in used_lid and (x1 - mx) <= lc['2d'][0] <= (x2 + mx) and (y1 - my) <= lc['2d'][1] <= (y2 + my):
                    used_lid.add(i)
                    # If Point-in-Box failed due to sparsity, fallback to the DBSCAN cluster center
                    if dist == 0.0:
                        dist, matched_lid_pts, source = lc['pos'][0], lc['pts'], "Fus" 
            
            for i, rp in enumerate(r_points):
                if i not in used_rad and (x1 - mx) <= rp['2d'][0] <= (x2 + mx) and (y1 - my) <= rp['2d'][1] <= (y2 + my):
                    rad_vel = rp['vel']
                    used_rad.add(i); source = "Rad+Fus" if source == "Fus" else "Rad+Cam"
                    if dist == 0.0: dist = rp['pos'][0]
                    break

            if dist < 0.5: 
                prior_h = 3.0 if d['label'] in ['truck', 'bus'] else (1.2 if d['label'] in ['motorcycle', 'bicycle'] else 1.5)
                dist = (self.K[1,1] * prior_h) / max(y2 - y1, 1)
            
            meas.append({'pos': [dist, (cx_px - self.K[0,2]) * dist / self.K[0,0]], 'label': d['label'], 'bbox': (x1,y1,x2,y2), 
                        'conf': d['conf'], 'source': source, 'lidar_pts': matched_lid_pts, 'rad_vel': rad_vel, 'features': d.get('features')})
        
        for i, lc in enumerate(l_clusters):
            if i in used_lid: continue 
            u, v = lc['2d']
            box = (int(max(0, u-25)), int(max(0, v-25)), int(min(self.width, u+25)), int(min(self.height, v+25)))
            
            rad_vel, source = None, "LiDAR"
            for j, rp in enumerate(r_points):
                if j not in used_rad and math.dist(lc['pos'], rp['pos']) < 3.0: 
                    rad_vel = rp['vel']; used_rad.add(j); source = "Rad+Lid"; break
            
            meas.append({'pos': lc['pos'], 'label': 'LIDAR_OBJ', 'bbox': box, 'conf': 0.8, 'source': source, 'lidar_pts': lc['pts'], 'rad_vel': rad_vel})
            
        for i, rp in enumerate(r_points):
            if i not in used_rad and abs(rp['vel']) >= 3.0:
                u, v = rp['2d']
                box = (int(max(0, u-15)), int(max(0, v-15)), int(min(self.width, u+15)), int(min(self.height, v+15)))
                meas.append({'pos': rp['pos'], 'label': 'MOVING_RADAR', 'bbox': box, 'conf': 0.3, 'source': "Radar", 'lidar_pts': 0, 'rad_vel': rp['vel']})

        filtered_tracks = [t for t in self.tracker.update(meas, ego_speed) if not (t.sensor_source in ["Camera", "Radar", "LiDAR"] and t.hit_streak < 4) and abs(t.velocity_vec[1]) <= 10.0]
        return filtered_tracks, self._bev(l3d, filtered_tracks), l3d

    def _cam(self, img_array):
        out = []
        for r in self.model(img_array, verbose=False, classes=[1,2,3,5,7], conf=0.30):
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                if y2 < self.height * 0.35: continue
                crop = img_array[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)]
                features = cv2.normalize(cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]), None).flatten() if crop.size > 0 else None
                out.append({"box": (x1, y1, x2, y2), "label": self.model.names[int(b.cls[0])], "conf": float(b.conf[0]), "features": features})
        return out

    def _lid_independent(self, data):
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy(); pts[:, 1] = -pts[:, 1] 
        if len(pts) == 0: return [], pts[::4]
        mask = (pts[:, 0] > 1.5) & (pts[:, 0] < 45.0) & (np.abs(pts[:, 1]) < 8.0) & (pts[:, 2] > -1.5) & (pts[:, 2] < 1.5)
        roi = pts[mask]
        clusters = []
        if len(roi) > 10:
            labels = DBSCAN(eps=2.0, min_samples=8).fit(roi[:, :2]).labels_
            for lbl in set(labels) - {-1}:
                obj = roi[labels == lbl]
                if 0.6 <= (np.max(obj[:,0]) - np.min(obj[:,0])) <= 8.0 and 0.6 <= (np.max(obj[:,1]) - np.min(obj[:,1])) <= 8.0:
                    cx, cy, cz = map(float, np.mean(obj[:, :3], axis=0))
                    u, v = (self.K[0,0] * -cy / cx) + self.K[0,2], (self.K[0,0] * -cz / cx) + self.K[1,2]
                    if 0 <= u < self.width and 0 <= v < self.height: clusters.append({'pos': [cx, cy], '2d': [int(u), int(v)], 'pts': len(obj)})
        return clusters, pts[::4]

    def _rad_independent(self, data):
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        if len(pts) == 0: return []
        x, y, z, vel = pts[:,0]*np.cos(pts[:,1])*np.cos(pts[:,2]), pts[:,0]*np.cos(pts[:,1])*np.sin(pts[:,2]), pts[:,0]*np.sin(pts[:,1]), pts[:,3]
        m = (x > 1.5) & (x < 45) & (abs(y) < 8.0)
        return [{'pos': [float(xi), float(yi)], '2d': [int((self.K[0,0] * -yi / xi) + self.K[0,2]), int((self.K[0,0] * -zi / xi) + self.K[1,2])], 'vel': float(vi)} 
                for xi, yi, zi, vi in zip(x[m], y[m], z[m], vel[m]) if 0 <= int((self.K[0,0] * -yi / xi) + self.K[0,2]) < self.width and 0 <= int((self.K[0,0] * -zi / xi) + self.K[1,2]) < self.height]

    def _bev(self, pts, objs):
        bev = np.zeros((600, 400, 3), dtype=np.uint8); scale, cx, cy = 6.0, 200, 550
        if len(pts) > 0:
            px, py = (cx + pts[:,1] * scale).astype(int), (cy - pts[:,0] * scale).astype(int)
            valid = (px >= 0) & (px < 400) & (py >= 0) & (py < 600)
            bev[py[valid], px[valid]] = (255, 255, 255)
        cv2.arrowedLine(bev, (cx, cy+10), (cx, cy-10), (0,0,255), 2)
        for o in objs:
            mx, my = int(cx - o.position[1]*scale), int(cy - o.position[0]*scale)
            if 0 <= mx < 400 and 0 <= my < 600:
                c = (0, 255, 0) if "Fus" in o.sensor_source else (0, 255, 255)
                cv2.circle(bev, (mx,my), 6, c, -1)
                cv2.line(bev, (mx,my), (int(mx - math.sin(math.radians(o.yaw))*15), int(my - math.cos(math.radians(o.yaw))*15)), c, 2)
                cv2.putText(bev, str(o.id), (mx+5, my), 0, 0.5, c, 1)
        return bev

def main():
    client = carla.Client('localhost', 2000); client.set_timeout(10.0) 
    world, tm = client.get_world(), client.get_trafficmanager()
    tm.set_synchronous_mode(False)
    
    batch = [carla.command.DestroyActor(x) for x in list(world.get_actors().filter('vehicle.*')) + list(world.get_actors().filter('sensor.*'))]
    if batch: client.apply_batch_sync(batch); time.sleep(1.0)
    
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 1/30.0
    world.apply_settings(settings); tm.set_synchronous_mode(True)
    
    bp_lib, sp = world.get_blueprint_library(), world.get_map().get_spawn_points()
    ego = next((world.try_spawn_actor(bp_lib.filter('vehicle.tesla.model3')[0], sp[i]) for i in range(15)), None)
    if not ego: return
    ego.set_autopilot(True)
    
    npcs, sp_shuffled = [], sp[:]
    random.shuffle(sp_shuffled)
    for p in sp_shuffled[:30]:
        npc = world.try_spawn_actor(random.choice(bp_lib.filter('vehicle.*')), p)
        if npc: npc.set_autopilot(True, tm.get_port()); npcs.append(npc.id)

    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640'); cam_bp.set_attribute('image_size_y', '480'); cam_bp.set_attribute('fov', '90')
    lid_bp = bp_lib.find('sensor.lidar.ray_cast')
    lid_bp.set_attribute('range', '80'); lid_bp.set_attribute('points_per_second', '3000000'); lid_bp.set_attribute('channels', '64'); lid_bp.set_attribute('rotation_frequency', '30')
    rad_bp = bp_lib.find('sensor.other.radar')
    rad_bp.set_attribute('horizontal_fov', '30'); rad_bp.set_attribute('vertical_fov', '15'); rad_bp.set_attribute('range', '80')

    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego)
    lid = world.spawn_actor(lid_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego)
    rad = world.spawn_actor(rad_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego)

    q_c, q_l, q_r = queue.Queue(), queue.Queue(), queue.Queue()
    cam.listen(q_c.put); lid.listen(q_l.put); rad.listen(q_r.put)
    
    eng, evaluator = SensorFusionEngine(640, 480, 90), MetricsTracker()
    print(">>> RUNNING FULL OBJECT-LEVEL FUSION (Redundant System) <<<")
    t_start, frames, fps = time.time(), 0, 0
    
    try:
        while True:
            world.tick()
            try: c_data, l_data, r_data = q_c.get(timeout=2.0), q_l.get(timeout=2.0), q_r.get(timeout=2.0)
            except queue.Empty: continue
            
            v = ego.get_velocity()
            ego_speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            tracks, bev, l3d = eng.process(c_data, l_data, r_data, ego_speed)
            
            sensor_matrix = np.array(cam.get_transform().get_inverse_matrix())
            gt_objects = []
            
            for npc_id in npcs:
                actor = world.get_actor(npc_id)
                if not actor: continue
                loc, bb = actor.get_location(), actor.bounding_box.extent
                local_pos = sensor_matrix @ np.array([loc.x, loc.y, loc.z, 1.0])
                if 2.0 < local_pos[0] < 45.0 and abs(local_pos[1]) < (local_pos[0] * 0.7): 
                    gt_objects.append({'id': npc_id, 'x': local_pos[0], 'y': local_pos[1], 'length': bb.x * 2.0, 'width': bb.y * 2.0})
                    
            evaluator.evaluate_frame(tracks, gt_objects)
            
            frames += 1
            if time.time() - t_start >= 1.0: fps, frames, t_start = frames, 0, time.time()
                
            print(f"\n--- Environment State (FPS: {fps}) ---\n{'ID':<4}|{'Class':<8}|{'Conf':<5}|{'Dx(m)':<6}|{'Dy(m)':<6}|{'V_Rel':<6}|{'REAL Spd':<8}|{'Src'}\n{'-'*65}")
            for t in tracks:
                if t.missed_frames <= 3:
                    r_spd = math.sqrt((t.velocity_vec[0] + ego_speed)**2 + t.velocity_vec[1]**2) * 3.6
                    print(f"{t.id:<4}|{t.class_label:<8}|{t.confidence:<4.2f}|{t.position[0]:<6.1f}|{t.position[1]:<6.1f}|{t.velocity_vec[0]:<6.1f}|{r_spd if r_spd >= 5.0 else 0.0:<8.1f}|{t.sensor_source}")

            img = np.frombuffer(c_data.raw_data, dtype=np.uint8).reshape(480, 640, 4)[:,:,:3].copy()
            for p in l3d[::2]: 
                if p[0] > 1.0: 
                    u, v = int((eng.K[0,0] * -p[1] / p[0]) + eng.K[0,2]), int((eng.K[0,0] * -p[2] / p[0]) + eng.K[1,2])
                    if 0 <= u < 640 and 0 <= v < 480: cv2.circle(img, (u, v), 1, (200, 200, 200), -1)

            for t in [t for t in tracks if t.missed_frames <= 3]:
                x1, y1, x2, y2 = t.bbox
                c = (0,255,0) if "Fus" in t.sensor_source or "Rad+Lid" in t.sensor_source else ((255,0,0) if "LiDAR" in t.sensor_source else ((0,255,255) if "Radar" in t.sensor_source else (0,0,255)))
                r_spd = math.sqrt((t.velocity_vec[0] + ego_speed)**2 + t.velocity_vec[1]**2) * 3.6
                cv2.rectangle(img, (x1,y1), (x2,y2), c, 2)
                cv2.putText(img, f"ID:{t.id} {math.hypot(*t.position):.1f}m {r_spd if r_spd>=5.0 else 0.0:.0f}km/h", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            
            cv2.putText(img, f"FPS: {fps}", (10,30), 0, 1, (0,255,255), 2)
            cv2.imshow("Redundant Late Fusion", np.hstack((img, cv2.resize(bev, (400, 480)))))
            if cv2.waitKey(1) == ord('q'): break
            
    finally:
        evaluator.print_summary(); evaluator.plot_metrics() 
        settings.synchronous_mode = False; world.apply_settings(settings)
        for actor in filter(None, [ego, cam, lid, rad] + [world.get_actor(x) for x in npcs]): actor.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__": main()