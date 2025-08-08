
from rhos_cobot.utils import load_hdf5

def replay_joint_data(args):
    rospy.init_node("replay_node")
    bridge = CvBridge()
    img_left_publisher = rospy.Publisher(
        args.img_left_topic, Image, queue_size=10)
    img_right_publisher = rospy.Publisher(
        args.img_right_topic, Image, queue_size=10)
    img_front_publisher = rospy.Publisher(
        args.img_front_topic, Image, queue_size=10)

    puppet_arm_left_publisher = rospy.Publisher(
        args.puppet_arm_left_topic, JointState, queue_size=10)
    puppet_arm_right_publisher = rospy.Publisher(
        args.puppet_arm_right_topic, JointState, queue_size=10)

    master_arm_left_publisher = rospy.Publisher(
        args.master_arm_left_topic, JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(
        args.master_arm_right_topic, JointState, queue_size=10)

    robot_base_publisher = rospy.Publisher(
        args.robot_base_topic, Twist, queue_size=10)

    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    task_name = args.task_name
    dataset_name = f'episode_{episode_idx}'

    origin_left = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279]
    origin_right = [0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]

    joint_state_msg = JointState()
    joint_state_msg.header = Header()
    joint_state_msg.name = ['joint0', 'joint1', 'joint2',
                            'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
    twist_msg = Twist()

    rate = rospy.Rate(args.frame_rate)

    qposs, qvels, efforts, actions, base_actions, image_dicts, _ = load_hdf5(
        os.path.join(dataset_dir, task_name), dataset_name)

    if args.only_pub_master:
        last_action = [-0.0057, -0.031, -0.0122, -0.032, 0.0099, 0.0179,
                       0.2279, 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
        rate = rospy.Rate(100)
        for action in actions:
            if (rospy.is_shutdown()):
                break

            new_actions = np.linspace(last_action, action, 20)  # 插值
            last_action = action
            for act in new_actions:
                print(np.round(act, 4))
                cur_timestamp = rospy.Time.now()  # 设置时间戳
                joint_state_msg.header.stamp = cur_timestamp

                joint_state_msg.position = act[:7]
                master_arm_left_publisher.publish(joint_state_msg)

                joint_state_msg.position = act[7:]
                master_arm_right_publisher.publish(joint_state_msg)

                if (rospy.is_shutdown()):
                    break
                rate.sleep()

    else:
        i = 0
        while (not rospy.is_shutdown() and i < len(actions)):
            print("left: ", np.round(qposs[i][:7], 4),
                  " right: ", np.round(qposs[i][7:], 4))

            cam_names = [k for k in image_dicts.keys()]
            image0 = image_dicts[cam_names[0]][i]
            image0 = cv2.imdecode(np.frombuffer(
                image0, np.uint8), cv2.IMREAD_COLOR)
            image0 = image0[:, :, [2, 1, 0]]  # swap B and R channel

            image1 = image_dicts[cam_names[1]][i]
            image1 = cv2.imdecode(np.frombuffer(
                image1, np.uint8), cv2.IMREAD_COLOR)
            image1 = image1[:, :, [2, 1, 0]]  # swap B and R channel

            image2 = image_dicts[cam_names[2]][i]
            image2 = cv2.imdecode(np.frombuffer(
                image2, np.uint8), cv2.IMREAD_COLOR)
            image2 = image2[:, :, [2, 1, 0]]  # swap B and R channel

            cur_timestamp = rospy.Time.now()  # 设置时间戳

            joint_state_msg.header.stamp = cur_timestamp
            joint_state_msg.position = actions[i][:7]
            master_arm_left_publisher.publish(joint_state_msg)

            joint_state_msg.position = actions[i][7:]
            master_arm_right_publisher.publish(joint_state_msg)

            joint_state_msg.position = qposs[i][:7]
            puppet_arm_left_publisher.publish(joint_state_msg)

            joint_state_msg.position = qposs[i][7:]
            puppet_arm_right_publisher.publish(joint_state_msg)

            img_front_publisher.publish(bridge.cv2_to_imgmsg(image0, "bgr8"))
            img_left_publisher.publish(bridge.cv2_to_imgmsg(image1, "bgr8"))
            img_right_publisher.publish(bridge.cv2_to_imgmsg(image2, "bgr8"))

            twist_msg.linear.x = base_actions[i][0]
            twist_msg.angular.z = base_actions[i][1]
            robot_base_publisher.publish(twist_msg)

            i += 1
            rate.sleep()