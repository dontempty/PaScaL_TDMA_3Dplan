## Compile

```
make lib
make example
make all
```

```
cd run

```

원래는 solve_many를 n_sys만큼 불러서 2차워 plan을 여러변 풀고 있었는데
여기서는 3D plan을 만들어서 함수 호출을 한 번만 하자.

결과: 더 안좋아짐

//
examples/backup/solve_theta_2D가 옛날 버전입니다.
