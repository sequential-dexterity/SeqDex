link = """
  <link name="%(refID)s">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <mesh filename="%(mesh)s" scale="%(m_scale)s %(m_scale)s %(m_scale)s"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <mesh filename="%(mesh)s" scale="%(m_scale)s %(m_scale)s %(m_scale)s"/>
      </geometry>
    </collision>
    <inertial>
      <density value="567.0"/>
    </inertial>
  </link>
"""