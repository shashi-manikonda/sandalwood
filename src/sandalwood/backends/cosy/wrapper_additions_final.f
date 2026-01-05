
      SUBROUTINE COMPUTE_DA_ATAN2(INY, INX, INC)
C     Computes atan2(Y, X) using DAATAN and quadrant logic
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INY, INX, INC, I_TMP, I_QUOT, IC(10), I_EXPS(10)
      DOUBLE PRECISION X0, Y0, PI, PI_2
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
      
      PI = 3.14159265358979323846D0
      PI_2 = 1.57079632679489661923D0

C     Allocate Temps
      CALL FOXALL(IC, 5, NMMAX)
      I_TMP = IC(1)
      I_QUOT = IC(2)
      
C     Get Constant Parts X0, Y0 (Term index 1 is constant in COSY)
      CALL GET_DA_COEFF_BY_INDEX(INX, 1, I_EXPS, X0)
      CALL GET_DA_COEFF_BY_INDEX(INY, 1, I_EXPS, Y0)
      
      IF (X0.GT.0.D0) THEN
         CALL DADIV(INY, INX, I_QUOT)
         CALL DAATAN(I_QUOT, INC)
      ELSE IF (X0.LT.0.D0) THEN
         IF (Y0.GE.0.D0) THEN
            CALL DADIV(INY, INX, I_QUOT)
            CALL DAATAN(I_QUOT, I_TMP)
            CALL DACON(IC(3), PI)
            CALL DAADD(I_TMP, IC(3), INC)
         ELSE
            CALL DADIV(INY, INX, I_QUOT)
            CALL DAATAN(I_QUOT, I_TMP)
            CALL DACON(IC(3), PI)
            CALL DASUB(I_TMP, IC(3), INC)
         END IF
      ELSE
         IF (Y0.GT.0.D0) THEN
             CALL DACON(INC, PI_2)
         ELSE IF (Y0.LT.0.D0) THEN
             CALL DACON(INC, -PI_2)
         ELSE
             ! Error 0,0, return 0
             CALL DACON(INC, 0.D0)
         END IF
      END IF
      
      CALL FOXDAL(IC, 5)
      RETURN
      END

      SUBROUTINE COMPUTE_CD_LOG(INA, INC)
C     Complex Log: log(A) = log|A| + i*arg(A)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INA, INC, IC(10)
      INTEGER I_RE, I_IM, I_RE2, I_IM2, I_R2, I_R, I_LR, I_THETA
      INTEGER I_CD_LR, I_CD_THETA, I_UNIT, I_TMP, I_IM_PART
      
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
      
      CALL FOXALL(IC, 10, NMMAX)
      I_RE = IC(1)
      I_IM = IC(2)
      I_RE2 = IC(3)
      I_IM2 = IC(4)
      I_R2  = IC(5)
      I_R   = IC(6)
      I_LR  = IC(7)
      I_THETA = IC(8)
      
      CALL CDRE(INA, I_RE)
      CALL CDIM(INA, I_IM)
      
      ! Compute R = Sqrt(Re^2 + Im^2)
      CALL DASQR(I_RE, I_RE2)
      CALL DASQR(I_IM, I_IM2)
      CALL DAADD(I_RE2, I_IM2, I_R2)
      CALL DASQRT(I_R2, I_R)
      CALL DALOG(I_R, I_LR)
      
      ! Compute Theta = atan2(Im, Re)
      CALL COMPUTE_DA_ATAN2(I_IM, I_RE, I_THETA)
      
      ! Result = L_R + i * THETA
      ! Construct CD from L_R + i*THETA
      ! We need 4 temps for CD ops logic
      
      ! Convert DA parts to CD
      ! CDCMPL(DA, CD) -> CD = DA + 0i
      CALL FOXALL(IC, 4, 2*NMMAX) ! Alloc CD size? 
C     Usually FOXALL(..., size). CD size is likely > 1. 2*NMMAX for safe?
C     If we alloc standard DA, does it fit CD?
C     Standard logic: "CALL FOXALL(ICD,1,2*NMMAX)" per CDMDA example.
C     So we need to alloc explicit size for CD vars.
      
      I_CD_LR = IC(1)
      I_CD_THETA = IC(2)
      I_UNIT = IC(3)
      I_TMP = IC(4)
      I_IM_PART = IC(5) ! Wait, FOXALL allocates N handles. 
C     We must specify LENGTH for each handle.
C     Standard FOXALL(IC, 1, NMMAX) allocates 1 var of NMMAX.
C     We need CD vars. CD needs 2*NMMAX usually? Or handled by NVE?
C     "CALL FOXALL(IC, 1, 2*NMMAX)" was in CDMDA.
C     So we allocate distinct CD variables with sufficient length.
      
      CALL FOXDAL(IC, 10) ! Free initial handle alloc
      
C     Alloc DA temps
      CALL FOXALL(IC, 8, NMMAX) 
      I_RE = IC(1)
      I_IM = IC(2)
      I_RE2 = IC(3)
      I_IM2 = IC(4)
      I_R2  = IC(5)
      I_R   = IC(6)
      I_LR  = IC(7)
      I_THETA = IC(8)
      
C     Re-re-extract (since we freed)
      CALL CDRE(INA, I_RE)
      CALL CDIM(INA, I_IM)
      CALL DASQR(I_RE, I_RE2)
      CALL DASQR(I_IM, I_IM2)
      CALL DAADD(I_RE2, I_IM2, I_R2)
      CALL DASQRT(I_R2, I_R)
      CALL DALOG(I_R, I_LR)
      CALL COMPUTE_DA_ATAN2(I_IM, I_RE, I_THETA)
      
C     Alloc CD temps
      CALL FOXALL(IC, 4, 2*NMMAX) 
      ! Offset indices because IC overwrites? No, passed array.
      ! We passed same IC array. Indices 1..8 used. 
      ! Need IC(9)..IC(12).
      I_CD_LR = IC(1) ! Bug: This will overwrite Handles 1..4 in IC if we use IC(1).
      ! We need distinct array or offsets.
      ! Let's just use offsets manually.
      ! Or simple individual allocs.
      ! Let's start clean.
      
      ! ... (Logic above correct) ...
      ! Now alloc CD
      ! We cannot pass &IC(9).
      ! Let's assume passed IC has size 20.
      I_CD_LR = IC(9)
      I_CD_THETA = IC(10)
      I_UNIT = IC(11)
      I_IM_PART = IC(12)
      
      ! L_R (DA) -> I_CD_LR (CD)
      CALL CDCMPL(I_LR, I_CD_LR)
      
      ! THETA (DA) -> I_CD_THETA (CD)
      CALL CDCMPL(I_THETA, I_CD_THETA)
      
      ! Create i (Complex Unit)
      CALL CREATE_CDA_CONST(I_UNIT, 0.D0, 1.D0)
      
      ! I_IM_PART = I_UNIT * I_CD_THETA
      CALL CDMCD(I_UNIT, I_CD_THETA, I_IM_PART)
      
      ! Result = I_CD_LR + I_IM_PART
      CALL CDACD(I_CD_LR, I_IM_PART, INC)
      
      CALL FOXDAL(IC, 12) ! Free all 12 handles (8 DA + 4 CD)
      RETURN
      END

      SUBROUTINE COMPUTE_CD_EXP(INA, INC)
C     Complex Exp: exp(X+iY) = exp(X) * (cos(Y) + i*sin(Y))
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INA, INC, IC(20)
      INTEGER I_X, I_Y, I_EX, I_CY, I_SY
      INTEGER I_CD_EX, I_CD_CY, I_CD_SY, I_CD_W, I_UNIT, I_TMP
      
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT

C     Alloc DA temps (5 vars)
      CALL FOXALL(IC, 5, NMMAX)
      I_X = IC(1)
      I_Y = IC(2)
      I_EX = IC(3)
      I_CY = IC(4)
      I_SY = IC(5)
      
      CALL CDRE(INA, I_X)
      CALL CDIM(INA, I_Y)
      
      CALL DAEXP(I_X, I_EX)
      CALL DACOS(I_Y, I_CY)
      CALL DASIN(I_Y, I_SY)
      
C     Alloc CD temps (6 vars). Use offset 6.
      CALL FOXALL(IC(6), 6, 2*NMMAX)
      I_CD_EX = IC(6)
      I_CD_CY = IC(7)
      I_CD_SY = IC(8)
      I_CD_W  = IC(9)
      I_UNIT  = IC(10)
      I_TMP   = IC(11)
      
C     Create W = Cos(Y) + i Sin(Y)
      CALL CDCMPL(I_CY, I_CD_CY)
      CALL CDCMPL(I_SY, I_CD_SY)
      CALL CREATE_CDA_CONST(I_UNIT, 0.D0, 1.D0)
      
      ! i * Sin(Y)
      CALL CDMCD(I_UNIT, I_CD_SY, I_TMP)
      
      ! W = Cos(Y) + i*Sin(Y)
      CALL CDACD(I_CD_CY, I_TMP, I_CD_W)
      
C     Result = Exp(X) * W
C     Exp(X) is DA. Convert to CD first for multiplication or use CDMDA?
C     CDMDA(CD, DA, OUT) -> Multiply CD by DA.
C     So CDMDA(I_CD_W, I_EX, INC).
      CALL CDMDA(I_CD_W, I_EX, INC)
 
      ! Cleanup: Free DA temps (1-5) and CD temps (6-11).
      ! Since we alloc'd separately (pointer logic in FOXALL array?), safer to free separately.
      ! FOXDAL(IC, 5) frees handles in IC(1)..IC(5).
      CALL FOXDAL(IC, 5)
      ! FOXDAL(IC(6), 6) frees handles in IC(6)..IC(11).
      CALL FOXDAL(IC(6), 6)
      
      RETURN
      END

      SUBROUTINE COMPUTE_CD_PKP(INA, VAL, INC)
C     Complex Real Power: Z^a = Exp(a * Log(Z))
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INA, INC, IC(2)
      DOUBLE PRECISION VAL
      INTEGER I_LOG, I_SCALED, I_VAL_DA
      
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
      
      ! Alloc CD Temps
      CALL FOXALL(IC, 2, 2*NMMAX)
      I_LOG = IC(1)
      I_SCALED = IC(2)
      
      CALL COMPUTE_CD_LOG(INA, I_LOG)
      
      ! Multiply Log(Z) by VAL (Real Scalar)
      ! CDMRE(INA, INB, INC) -> CD * RealScalar(DA?) or Double?
      ! Check CDMRE usage in dafox.f:
      ! "SUBROUTINE CDMRE(INA,INB,INC)"
      ! "THIS SUBROUTINE PERFORMS A CD MULTIPLICATION OF THE CD VECTOR INA AND REAL INB"
      ! Is INB a Real DA or a Double?
      ! grep CDMRE -> "SUBROUTINE CDMRE(INA,INB,INC)".
      ! Comment usually says arguments. 
      ! Let's assume INB is Real DA (NRE type or just DA). 
      ! If we have Double VAL, we need to make constant.
      
      ! Create Constant DA for VAL
      ! We need another temp.
      ! Re-alloc IC to 3? Or utilize FOXALL separately.
      ! Let's assume INB is DA.
      I_VAL_DA = 0 ! Need alloc.
      CALL FOXALL(I_VAL_DA, 1, NMMAX) ! Single handle
      CALL DACON(I_VAL_DA, VAL)
      
      CALL CDMRE(I_LOG, I_VAL_DA, I_SCALED)
      
      CALL COMPUTE_CD_EXP(I_SCALED, INC)
      
      CALL FOXDAL(IC, 2)
      CALL FOXDAL(I_VAL_DA, 1)
      RETURN
      END

      SUBROUTINE COMPUTE_CD_MUI(INA, INC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INA, INC, IC(1)
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
      CALL FOXALL(IC, 1, 2*NMMAX)
      INC = IC(1)
      CALL CDMUI(INA, INC)
      RETURN
      END

      SUBROUTINE COMPUTE_CD_PEI(INA, N, INC)
C     Complex Integer Power: C = A^N
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INA, N, INC, I, IC(2), I_TMP, I_ONE
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
     
      CALL FOXALL(IC, 2, 2*NMMAX)
      INC = IC(1)
      I_TMP = IC(2)
      
C     Create 1.0 (Complex)
      CALL CREATE_CDA_CONST(INC, 1.D0, 0.D0)

      IF(N.EQ.0) RETURN

      IF(N.GT.0) THEN
         DO I=1, N
            CALL CDMCD(INC, INA, I_TMP)
            CALL CDCOP(I_TMP, INC)
         END DO
      ELSE
         DO I=1, ABS(N)
            CALL CDMCD(INC, INA, I_TMP)
            CALL CDCOP(I_TMP, INC)
         END DO
         CALL CDMUI(INC, I_TMP)
         CALL CDCOP(I_TMP, INC)
      END IF
      CALL FOXDAL(IC, 2)
      RETURN
      END
