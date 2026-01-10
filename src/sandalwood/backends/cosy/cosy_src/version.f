      PROGRAM VERSION
*     ***************
*
*     THIS PROGRAM AUTOMATICALLY CHANGES A CODE TO RUN ON A DIFFERENT
*     COMPUTER BY COMMENTING OFF SOME LINES AND UNCOMMENTING OTHERS.
*     THE USER SPECIFIES A CHARACTER STRING WHICH, WHEN FOUND IN COLUMNS
*     73:80, ENTAILS THAT THE LINE BE COMMENTED OUT. ANOTHER STRING IS
*     SUPPLIED WHICH, WHEN FOUND STRARTING IN COLUMN 1, ENTAILS THAT
*     THE LINE IS UNCOMMENTED.
*
      CHARACTER A*256,S1*10,S2*10,FILE1*256,FILE2*256,BL*5
      DATA  BL / '     ' /
*
      PRINT*,'    ****************************************************'
      PRINT*,'    *                                                  *'
      PRINT*,'    *             UTILITY PROGRAM  VERSION             *'
      PRINT*,'    *                                                  *'
      PRINT*,'    * This program changes the type of machine/system. *'
      PRINT*,'    * The current COSY INFINITY system supports        *'
      PRINT*,'    * NORM MPI FACE RND, PGP GRW AQT, and IFOR GFOR.   *'
      PRINT*,'    * See the User''s Guide and Reference Manual.       *'
      PRINT*,'    *                                                  *'
      PRINT*,'    ****************************************************'
      PRINT*,' '
      PRINT*,'GIVE OLD FILENAME:'
      READ(*,'(A)') FILE1
      PRINT*,' '
      PRINT*,'GIVE NEW FILENAME:'
      READ(*,'(A)') FILE2
*
      PRINT*,' '
      PRINT*,'SPECIFY ID OF CURRENT VERSION (MUST START WITH * OR C):'
      PRINT*,
     *'Examples: *PGP *GRW *AQT, *NORM *MPI *FACE *RND, and *IFOR *GFOR'
      PRINT*,' '
      S1 = BL
      READ(*,'(A)') S1
      DO 1 L1=5,1,-1
      IF(S1(L1:L1).NE.' ') GOTO 2
  1   CONTINUE
  2   CONTINUE
*
      PRINT*,' '
      PRINT*,'SPECIFY ID OF NEW VERSION (MUST START WITH * OR C):'
      PRINT*,' '
      S2 = BL
      READ(*,'(A)') S2
      DO 3 L2=5,1,-1
      IF(S2(L2:L2).NE.' ') GOTO 4
  3   CONTINUE
  4   CONTINUE
*
      OPEN(1,FILE=FILE1,STATUS='OLD')
      OPEN(2,FILE=FILE2,STATUS='UNKNOWN')
*
  10  CONTINUE
*
      READ(1,'(A)',END=100) A
      DO 20 IA=80,1,-1
      IF(A(IA:IA).NE.' ') GOTO 30
  20  CONTINUE
  30  CONTINUE
      IF(INDEX(A(73:80),S1(1:L1)).NE.0)     A(1:L1) = S1(1:L1)
      IF(A(1:L2).EQ.S2(1:L2))               A(1:L2) = BL(1:L2)
      WRITE(2,'(A)') A(1:IA)
*
      GOTO 10
*
 100  CONTINUE
      CLOSE(1)
      CLOSE(2)
      PRINT*,'The VERSION change finished.'
      STOP
      END
